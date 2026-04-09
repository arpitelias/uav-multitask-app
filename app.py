import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import swin_t,Swin_T_Weights
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2

st.set_page_config(
    page_title="UAV Multi-Task Interpretation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""",unsafe_allow_html=True)

SEG_CLASSES=['building','road','tree','low_veg','moving_car','static_car','human','clutter']
SEG_COLORS=np.array([
    [128,0,0],
    [128,64,128],
    [0,128,0],
    [128,128,0],
    [64,0,128],
    [192,0,192],
    [64,64,0],
    [0,0,0]
])

SCENE_CLASSES=['mixed','sparse','traffic','urban-pedestrian','two-wheeler']

MODEL_INFO={
    'Single-Task (ResNet-50)':{
        'file':'resnet50_scene.pth',
        'params':'23.5M',
        'speed':'5.72 ms',
        'description':'ResNet-50 trained independently for scene classification only.'
    },
    'Single-Task Segmentation (ResNet-50)':{
        'file':'resnet50_seg.pth',
        'params':'31.5M',
        'speed':'7.39 ms',
        'description':'ResNet-50 with U-Net decoder for pixel-level segmentation.'
    },
    'Standard MTL':{
        'file':'mtl_standard.pth',
        'params':'24.6M',
        'speed':'5.47 ms',
        'description':'Multi-task model with shared ResNet-50 backbone and separate heads for segmentation and classification.'
    },
    'Cross-Task Attention':{
        'file':'mtl_cross_attention.pth',
        'params':'25.7M',
        'speed':'6.43 ms',
        'description':'Novel multi-task model with cross-task attention module enabling task heads to share information.'
    },
    'Swin Transformer MTL':{
        'file':'swin_mtl.pth',
        'params':'27.8M',
        'speed':'8.21 ms',
        'description':'Multi-task model using Swin Transformer backbone with hierarchical shifted-window attention.'
    },
    '3-Task MTL':{
        'file':'three_task_mtl_best.pth',
        'params':'26.3M',
        'speed':'6.85 ms',
        'description':'Full 3-task multi-task model handling segmentation, classification, and detection simultaneously.'
    }
}


class MultiTaskModel(nn.Module):
    def __init__(self,num_seg_classes=8,num_scene_classes=5):
        super().__init__()
        backbone=models.resnet50(weights=None)
        self.encoder_layers=nn.Sequential(
            backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
            backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4
        )
        self.seg_up1=nn.ConvTranspose2d(2048,512,kernel_size=2,stride=2)
        self.seg_conv1=nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.ReLU(),nn.BatchNorm2d(512))
        self.seg_up2=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.seg_conv2=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.BatchNorm2d(256))
        self.seg_up3=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.seg_conv3=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128))
        self.seg_up4=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.seg_conv4=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64))
        self.seg_up5=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.seg_final=nn.Conv2d(32,num_seg_classes,1)
        self.cls_pool=nn.AdaptiveAvgPool2d(1)
        self.cls_fc=nn.Sequential(
            nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(512,num_scene_classes)
        )
        self.log_var_seg=nn.Parameter(torch.zeros(1))
        self.log_var_cls=nn.Parameter(torch.zeros(1))

    def forward(self,x,task='both'):
        features=self.encoder_layers(x)
        outputs={}
        if task in ['seg','both']:
            s=self.seg_up1(features)
            s=self.seg_conv1(s)
            s=self.seg_up2(s)
            s=self.seg_conv2(s)
            s=self.seg_up3(s)
            s=self.seg_conv3(s)
            s=self.seg_up4(s)
            s=self.seg_conv4(s)
            s=self.seg_up5(s)
            outputs['seg']=self.seg_final(s)
        if task in ['cls','both']:
            c=self.cls_pool(features).flatten(1)
            outputs['cls']=self.cls_fc(c)
        return outputs


class CrossTaskAttention(nn.Module):
    def __init__(self,feature_dim=512,num_heads=4):
        super().__init__()
        self.attention=nn.MultiheadAttention(embed_dim=feature_dim,num_heads=num_heads,batch_first=True)
        self.norm=nn.LayerNorm(feature_dim)
        self.proj=nn.Linear(feature_dim,feature_dim)

    def forward(self,query_feat,context_feat):
        b,c,h,w=query_feat.shape
        q=query_feat.flatten(2).permute(0,2,1)
        ctx=context_feat.flatten(2).permute(0,2,1)
        attended,_=self.attention(q,ctx,ctx)
        attended=self.norm(attended+q)
        attended=self.proj(attended)
        return attended.permute(0,2,1).reshape(b,c,h,w)


class MultiTaskCrossAttention(nn.Module):
    def __init__(self,num_seg_classes=8,num_scene_classes=5):
        super().__init__()
        backbone=models.resnet50(weights=None)
        self.encoder=nn.Sequential(
            backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
            backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4
        )
        self.seg_reduce=nn.Sequential(nn.Conv2d(2048,512,1),nn.ReLU())
        self.cls_reduce=nn.Sequential(nn.Conv2d(2048,512,1),nn.ReLU())
        self.seg_attend_cls=CrossTaskAttention(512,num_heads=4)
        self.cls_attend_seg=CrossTaskAttention(512,num_heads=4)
        self.seg_up1=nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.seg_c1=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.BatchNorm2d(256))
        self.seg_up2=nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.seg_c2=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128))
        self.seg_up3=nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.seg_c3=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64))
        self.seg_up4=nn.ConvTranspose2d(64,32,kernel_size=2,stride=2)
        self.seg_up5=nn.ConvTranspose2d(32,16,kernel_size=2,stride=2)
        self.seg_final=nn.Conv2d(16,num_seg_classes,1)
        self.cls_pool=nn.AdaptiveAvgPool2d(1)
        self.cls_head=nn.Sequential(
            nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,num_scene_classes)
        )
        self.log_var_seg=nn.Parameter(torch.zeros(1))
        self.log_var_cls=nn.Parameter(torch.zeros(1))

    def forward(self,x,task='both'):
        shared=self.encoder(x)
        outputs={}
        seg_feat=self.seg_reduce(shared)
        cls_feat=self.cls_reduce(shared)
        if task in ['seg','both']:
            enhanced_seg=self.seg_attend_cls(seg_feat,cls_feat)
            s=self.seg_up1(enhanced_seg)
            s=self.seg_c1(s)
            s=self.seg_up2(s)
            s=self.seg_c2(s)
            s=self.seg_up3(s)
            s=self.seg_c3(s)
            s=self.seg_up4(s)
            s=self.seg_up5(s)
            outputs['seg']=self.seg_final(s)
        if task in ['cls','both']:
            enhanced_cls=self.cls_attend_seg(cls_feat,seg_feat)
            c=self.cls_pool(enhanced_cls).flatten(1)
            outputs['cls']=self.cls_head(c)
        return outputs


class ThreeTaskMTL(nn.Module):
    def __init__(self,num_seg=8,num_cls=5):
        super().__init__()
        backbone=models.resnet50(weights=None)
        self.encoder=nn.Sequential(
            backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
            backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4)
        self.up1=nn.ConvTranspose2d(2048,512,2,stride=2)
        self.c1=nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.ReLU(),nn.BatchNorm2d(512))
        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.c2=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.BatchNorm2d(256))
        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.c3=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128))
        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.c4=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64))
        self.up5=nn.ConvTranspose2d(64,32,2,stride=2)
        self.seg_head=nn.Conv2d(32,num_seg,1)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.classifier=nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(512,num_cls))
        self.det_head=nn.Sequential(
            nn.Conv2d(2048,512,3,padding=1),nn.ReLU(),nn.BatchNorm2d(512),
            nn.Conv2d(512,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128),
            nn.Conv2d(128,1,1),nn.Sigmoid())
        self.log_var_seg=nn.Parameter(torch.zeros(1))
        self.log_var_cls=nn.Parameter(torch.zeros(1))
        self.log_var_det=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        f=self.encoder(x)
        s=self.c1(self.up1(f))
        s=self.c2(self.up2(s))
        s=self.c3(self.up3(s))
        s=self.c4(self.up4(s))
        s=self.up5(s)
        seg_out=self.seg_head(s)
        cls_out=self.classifier(self.pool(f).flatten(1))
        det_out=self.det_head(f)
        return seg_out,cls_out,det_out


HF_REPO='arpitjoshua/uav-multitask-models'

def download_from_hf(filename,local_path):
    import urllib.request
    url=f'https://huggingface.co/{HF_REPO}/resolve/main/{filename}'
    try:
        with st.spinner(f'Downloading {filename} from Hugging Face (first-time only)...'):
            urllib.request.urlretrieve(url,local_path)
        return True
    except Exception as e:
        st.error(f'Failed to download {filename}: {e}')
        return False


@st.cache_resource
def load_model(model_name):
    device=torch.device('cpu')
    weights_dir='models'
    os.makedirs(weights_dir,exist_ok=True)
    file_name=MODEL_INFO[model_name]['file']
    file_path=os.path.join(weights_dir,file_name)

    if not os.path.exists(file_path):
        success=download_from_hf(file_name,file_path)
        if not success:
            return None,False

    if model_name=='Single-Task (ResNet-50)':
        m=models.resnet50(weights=None)
        m.fc=nn.Linear(m.fc.in_features,5)
        if os.path.exists(file_path):
            m.load_state_dict(torch.load(file_path,map_location=device))
            loaded=True
        else:
            loaded=False
        m.eval()
        return m,loaded

    elif model_name=='Single-Task Segmentation (ResNet-50)':
        class SegModel(nn.Module):
            def __init__(self):
                super().__init__()
                backbone=models.resnet50(weights=None)
                self.enc=nn.Sequential(
                    backbone.conv1,backbone.bn1,backbone.relu,backbone.maxpool,
                    backbone.layer1,backbone.layer2,backbone.layer3,backbone.layer4)
                self.up1=nn.ConvTranspose2d(2048,512,2,stride=2)
                self.c1=nn.Sequential(nn.Conv2d(512,512,3,padding=1),nn.ReLU(),nn.BatchNorm2d(512))
                self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
                self.c2=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU(),nn.BatchNorm2d(256))
                self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
                self.c3=nn.Sequential(nn.Conv2d(128,128,3,padding=1),nn.ReLU(),nn.BatchNorm2d(128))
                self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
                self.c4=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.BatchNorm2d(64))
                self.up5=nn.ConvTranspose2d(64,32,2,stride=2)
                self.seg_head=nn.Conv2d(32,8,1)
            def forward(self,x):
                f=self.enc(x)
                x=self.c1(self.up1(f))
                x=self.c2(self.up2(x))
                x=self.c3(self.up3(x))
                x=self.c4(self.up4(x))
                x=self.up5(x)
                return self.seg_head(x)
        m=SegModel()
        if os.path.exists(file_path):
            try:
                m.load_state_dict(torch.load(file_path,map_location=device))
                loaded=True
            except:
                loaded=False
        else:
            loaded=False
        m.eval()
        return m,loaded

    elif model_name=='Standard MTL':
        m=MultiTaskModel()
        if os.path.exists(file_path):
            m.load_state_dict(torch.load(file_path,map_location=device))
            loaded=True
        else:
            loaded=False
        m.eval()
        return m,loaded

    elif model_name=='Cross-Task Attention':
        m=MultiTaskCrossAttention()
        if os.path.exists(file_path):
            m.load_state_dict(torch.load(file_path,map_location=device))
            loaded=True
        else:
            loaded=False
        m.eval()
        return m,loaded

    elif model_name=='3-Task MTL':
        m=ThreeTaskMTL()
        if os.path.exists(file_path):
            m.load_state_dict(torch.load(file_path,map_location=device))
            loaded=True
        else:
            loaded=False
        m.eval()
        return m,loaded

    else:
        return None,False


def preprocess_image(img_array,size=256):
    tfm=A.Compose([
        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])
    img_resized=cv2.resize(img_array,(size,size))
    return tfm(image=img_resized)['image'].unsqueeze(0),img_resized


def segmentation_to_rgb(seg_mask):
    h,w=seg_mask.shape
    rgb=np.zeros((h,w,3),dtype=np.uint8)
    for c in range(8):
        rgb[seg_mask==c]=SEG_COLORS[c]
    return rgb


def run_inference(model,model_name,img_tensor,img_resized):
    results={}
    start=time.time()

    with torch.no_grad():
        if model_name=='Single-Task (ResNet-50)':
            out=model(img_tensor)
            probs=F.softmax(out,dim=1)[0].numpy()
            results['cls_probs']=probs
            results['cls_pred']=int(out.argmax(1).item())

        elif model_name=='Single-Task Segmentation (ResNet-50)':
            out=model(img_tensor)
            out=F.interpolate(out,size=img_resized.shape[:2],mode='bilinear',align_corners=False)
            seg_pred=out.argmax(1)[0].numpy()
            results['seg_mask']=seg_pred

        elif model_name in ['Standard MTL','Cross-Task Attention']:
            out=model(img_tensor,task='both')
            seg_out=F.interpolate(out['seg'],size=img_resized.shape[:2],mode='bilinear',align_corners=False)
            seg_pred=seg_out.argmax(1)[0].numpy()
            results['seg_mask']=seg_pred
            probs=F.softmax(out['cls'],dim=1)[0].numpy()
            results['cls_probs']=probs
            results['cls_pred']=int(out['cls'].argmax(1).item())

        elif model_name=='3-Task MTL':
            seg_out,cls_out,det_out=model(img_tensor)
            seg_out=F.interpolate(seg_out,size=img_resized.shape[:2],mode='bilinear',align_corners=False)
            det_out=F.interpolate(det_out,size=img_resized.shape[:2],mode='bilinear',align_corners=False)
            seg_pred=seg_out.argmax(1)[0].numpy()
            results['seg_mask']=seg_pred
            probs=F.softmax(cls_out,dim=1)[0].numpy()
            results['cls_probs']=probs
            results['cls_pred']=int(cls_out.argmax(1).item())
            results['det_heatmap']=det_out[0,0].numpy()

    results['inference_time_ms']=(time.time()-start)*1000
    return results


st.markdown('<p class="main-header">🛰️ Attention-Guided Multi-Task UAV Interpretation</p>',unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unified deep learning system for drone imagery — semantic segmentation, scene classification, and object detection</p>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    mode=st.radio("Mode",['Single Model Inference','Model Comparison','About the Project'],label_visibility='collapsed')

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    - **UAVid:** 670 images, 8 semantic classes
    - **VisDrone-2019:** 7,019 images, 5 scene types
    - **Resolution:** 256×256 (inference)
    """)

    st.markdown("---")
    st.markdown("### 👤 Author")
    st.markdown("""
    **Arpit Joshua Elias**  
    Student ID: 24257567  
    MSc AI — National College of Ireland  
    Supervisor: Prof. Furqan Rustam
    """)


if mode=='Single Model Inference':
    col_ctrl,col_main=st.columns([1,3])

    with col_ctrl:
        st.markdown("### Model Selection")
        model_name=st.selectbox(
            "Choose a model",
            list(MODEL_INFO.keys()),
            label_visibility='collapsed'
        )

        info=MODEL_INFO[model_name]
        st.markdown(f"**{model_name}**")
        st.caption(info['description'])

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['params']}</div>
            <div class="metric-label">Parameters</div>
        </div>
        """,unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-value">{info['speed']}</div>
            <div class="metric-label">Inference Speed</div>
        </div>
        """,unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Input Image")
        source=st.radio("Image source",['Upload','Sample'],label_visibility='collapsed')

        uploaded_img=None
        if source=='Upload':
            uploaded=st.file_uploader("Drop an image",type=['jpg','jpeg','png'],label_visibility='collapsed')
            if uploaded:
                uploaded_img=np.array(Image.open(uploaded).convert('RGB'))
        else:
            samples_dir='samples'
            if os.path.exists(samples_dir):
                sample_files=sorted([f for f in os.listdir(samples_dir) if f.endswith(('.jpg','.png'))])
                if sample_files:
                    selected=st.selectbox("Pick a sample",sample_files)
                    uploaded_img=np.array(Image.open(os.path.join(samples_dir,selected)).convert('RGB'))
                else:
                    st.warning("No sample images found in samples/")
            else:
                st.warning("samples/ directory not found")

    with col_main:
        if uploaded_img is not None:
            model,loaded=load_model(model_name)

            if not loaded:
                st.warning(f"⚠️ Model weights not found. Displaying architecture with random weights — predictions will not be meaningful. Place `{MODEL_INFO[model_name]['file']}` in the `models/` directory.")

            img_tensor,img_resized=preprocess_image(uploaded_img)
            results=run_inference(model,model_name,img_tensor,img_resized)

            st.markdown(f"### Inference Results — {model_name}")
            st.caption(f"Processed in {results['inference_time_ms']:.2f} ms on CPU")

            num_outputs=sum([
                'seg_mask' in results,
                'cls_probs' in results,
                'det_heatmap' in results
            ])+1

            cols=st.columns(num_outputs)
            col_idx=0

            with cols[col_idx]:
                st.markdown("**Original Image**")
                st.image(img_resized,use_column_width=True)
                col_idx+=1

            if 'seg_mask' in results:
                with cols[col_idx]:
                    st.markdown("**Segmentation**")
                    seg_rgb=segmentation_to_rgb(results['seg_mask'])
                    overlay=cv2.addWeighted(img_resized,0.5,seg_rgb,0.5,0)
                    st.image(overlay,use_column_width=True)
                    col_idx+=1

            if 'det_heatmap' in results:
                with cols[col_idx]:
                    st.markdown("**Object Detection**")
                    heatmap=results['det_heatmap']
                    heatmap_norm=(heatmap-heatmap.min())/(heatmap.max()-heatmap.min()+1e-6)
                    heatmap_colored=cv2.applyColorMap((heatmap_norm*255).astype(np.uint8),cv2.COLORMAP_JET)
                    heatmap_colored=cv2.cvtColor(heatmap_colored,cv2.COLOR_BGR2RGB)
                    overlay=cv2.addWeighted(img_resized,0.6,heatmap_colored,0.4,0)
                    st.image(overlay,use_column_width=True)
                    col_idx+=1

            if 'cls_probs' in results:
                with cols[col_idx]:
                    st.markdown("**Scene Classification**")
                    fig,ax=plt.subplots(figsize=(4,3))
                    probs=results['cls_probs']
                    colors=['#667eea' if i==results['cls_pred'] else '#c0c0c0' for i in range(len(probs))]
                    ax.barh(SCENE_CLASSES,probs,color=colors)
                    ax.set_xlim(0,1)
                    ax.set_xlabel('Probability')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig,use_container_width=True)
                    st.markdown(f"**Predicted:** {SCENE_CLASSES[results['cls_pred']]} ({probs[results['cls_pred']]*100:.1f}%)")

            if 'seg_mask' in results:
                with st.expander("📊 Class Distribution in Segmentation"):
                    unique,counts=np.unique(results['seg_mask'],return_counts=True)
                    total=results['seg_mask'].size
                    fig,ax=plt.subplots(figsize=(10,3))
                    cls_pcts=[]
                    cls_labels=[]
                    cls_colors=[]
                    for u,c in zip(unique,counts):
                        cls_pcts.append(c/total*100)
                        cls_labels.append(SEG_CLASSES[u])
                        cls_colors.append(SEG_COLORS[u]/255)
                    ax.barh(cls_labels,cls_pcts,color=cls_colors)
                    ax.set_xlabel('Percentage of pixels')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
        else:
            st.info("👈 Select or upload an image from the sidebar to begin")

            st.markdown("### 🎯 What this app does")
            st.markdown("""
            This deployment application demonstrates the multi-task deep learning framework for UAV image interpretation developed in this project.
            
            Upload any drone image (or use the provided samples) and select from six different model variants to see:
            
            - **Semantic Segmentation** — pixel-level labelling of 8 urban classes
            - **Scene Classification** — categorising overall scene context
            - **Object Detection** — spatial heatmap of object locations
            - **Model Comparison** — run different models side by side
            """)


elif mode=='Model Comparison':
    st.markdown("### 🔬 Compare Two Models Side by Side")

    col1,col2=st.columns(2)
    with col1:
        model_a=st.selectbox("Model A",list(MODEL_INFO.keys()),index=2)
    with col2:
        model_b=st.selectbox("Model B",list(MODEL_INFO.keys()),index=3)

    st.markdown("---")

    source=st.radio("Image source",['Upload','Sample'],horizontal=True)
    uploaded_img=None
    if source=='Upload':
        uploaded=st.file_uploader("Upload a drone image",type=['jpg','jpeg','png'])
        if uploaded:
            uploaded_img=np.array(Image.open(uploaded).convert('RGB'))
    else:
        samples_dir='samples'
        if os.path.exists(samples_dir):
            sample_files=sorted([f for f in os.listdir(samples_dir) if f.endswith(('.jpg','.png'))])
            if sample_files:
                selected=st.selectbox("Pick a sample",sample_files)
                uploaded_img=np.array(Image.open(os.path.join(samples_dir,selected)).convert('RGB'))

    if uploaded_img is not None:
        model_a_obj,loaded_a=load_model(model_a)
        model_b_obj,loaded_b=load_model(model_b)

        img_tensor,img_resized=preprocess_image(uploaded_img)
        results_a=run_inference(model_a_obj,model_a,img_tensor,img_resized)
        results_b=run_inference(model_b_obj,model_b,img_tensor,img_resized)

        st.markdown("### Results")

        col_a,col_b=st.columns(2)

        for col,model_n,results,info in [(col_a,model_a,results_a,MODEL_INFO[model_a]),(col_b,model_b,results_b,MODEL_INFO[model_b])]:
            with col:
                st.markdown(f"#### {model_n}")
                st.caption(f"Params: {info['params']} | Speed: {info['speed']} | Inference: {results['inference_time_ms']:.1f}ms")

                st.image(img_resized,caption='Original',use_column_width=True)

                if 'seg_mask' in results:
                    seg_rgb=segmentation_to_rgb(results['seg_mask'])
                    overlay=cv2.addWeighted(img_resized,0.5,seg_rgb,0.5,0)
                    st.image(overlay,caption='Segmentation',use_column_width=True)

                if 'cls_probs' in results:
                    fig,ax=plt.subplots(figsize=(5,2.5))
                    probs=results['cls_probs']
                    colors=['#667eea' if i==results['cls_pred'] else '#c0c0c0' for i in range(len(probs))]
                    ax.barh(SCENE_CLASSES,probs,color=colors)
                    ax.set_xlim(0,1)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig,use_container_width=True)
                    st.markdown(f"**Prediction:** {SCENE_CLASSES[results['cls_pred']]} ({probs[results['cls_pred']]*100:.1f}%)")


else:
    st.markdown("## 📖 About This Project")

    st.markdown("""
    ### Attention-Guided Multi-Task Learning for Unified UAV Image Interpretation

    This project addresses a fundamental limitation in aerial image analysis: existing systems run separate models for each task (detection, segmentation, classification), ignoring valuable cross-task information and wasting computation. This research proposes a unified multi-task deep learning framework with a **novel cross-task attention module** that enables task heads to share information during inference.
    """)

    col1,col2,col3=st.columns(3)
    with col1:
        st.metric("Best Segmentation","35.8% mIoU","Det+Seg Pair")
    with col2:
        st.metric("Best Classification","61.3%","Swin MTL")
    with col3:
        st.metric("Best Detection","78.1% F1","Det+Seg Pair")

    st.markdown("---")

    st.markdown("### 🎯 Research Question")
    st.info("Can a multi-task deep learning architecture with cross-task attention achieve superior aggregate performance across semantic segmentation, object detection, and scene classification on UAV imagery, compared to standard multi-task learning, independently trained single-task models, and traditional ML baselines?")

    st.markdown("### 🧪 Key Findings")
    st.markdown("""
    1. **Cross-task attention significantly improves classification** (p=0.022) without significant segmentation loss (p=0.304)
    2. **Detection + Segmentation synergy** — joint training improves segmentation (35.8% mIoU) beyond single-task baseline (34.5%)
    3. **3-Task unified model** successfully handles all tasks simultaneously (31.8% mIoU, 59.9% acc, 73.7% det F1)
    4. **Computational savings** — Multi-task models save 61.6% FLOPs compared to running single-task models separately
    5. **Backbone matters** — ResNet-50 outperforms EfficientNet-B3; Swin Transformer achieves best overall balance
    """)

    st.markdown("### 📊 Complete Ablation Results")
    import pandas as pd
    results_df=pd.DataFrame({
        'Model':['Seg Only','Cls Only','Seg+Cls Pair','Det+Seg Pair','Det+Cls Pair','Standard MTL','Cross-Task Attn','Swin MTL','EfficientNet-B3 MTL','3-Task MTL'],
        'Seg mIoU':[0.3446,'—',0.3182,0.3583,'—',0.3141,0.2958,0.3267,0.2744,0.3176],
        'Cls Acc':['—',0.5712,0.5839,'—',0.5675,0.5675,0.5876,0.6131,0.5712,0.5985],
        'Det F1':['—','—','—',0.7814,0.6954,'—','—','—','—',0.7367]
    })
    st.dataframe(results_df,use_container_width=True,hide_index=True)

    st.markdown("### 🛠️ Technical Stack")
    st.markdown("""
    - **Frameworks:** PyTorch, torchvision, albumentations
    - **Architectures:** ResNet-50, EfficientNet-B3, Swin Transformer, Faster R-CNN
    - **Novel contribution:** Cross-task attention module with multi-head attention
    - **Training:** Kendall's uncertainty-weighted multi-task loss, two-phase warmup strategy
    - **Evaluation:** mIoU, mAP@50:95, per-class AP, confusion matrices, gradient cosine similarity
    - **Deployment:** Streamlit web application (this app)
    """)

    st.markdown("---")
    st.caption("Project submitted as part of the MSc Artificial Intelligence programme at National College of Ireland, 2026.")
