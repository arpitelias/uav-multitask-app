import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, time, urllib.request

st.set_page_config(
    page_title="UAV Multi-Task Interpretation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header { font-size: 2.3rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
.sub-header { font-size: 1rem; color: #555; margin-top: 0; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white;
    text-align: center; margin-bottom: 0.8rem;
}
.metric-val { font-size: 1.6rem; font-weight: 700; }
.metric-lbl { font-size: 0.8rem; opacity: 0.9; }
.stButton>button {
    background-color: #667eea; color: white; border-radius: 5px;
    border: none; padding: 0.5rem 1.5rem; font-weight: 600;
}
.stButton>button:hover { background-color: #764ba2; }
</style>
""", unsafe_allow_html=True)


SEG_CLASSES = ['building','road','tree','low_veg','moving_car','static_car','human','clutter']
SEG_COLORS = np.array([
    [128,128,0],
    [128,0,0],
    [0,128,0],
    [128,64,128],
    [192,0,192],
    [64,0,128],
    [64,64,0],
    [0,0,0]
], dtype=np.uint8)

SCENE_CLASSES = ['sparse','traffic','urban-pedestrian','two-wheeler','mixed']


MODEL_INFO = {
    'Single-Task Classifier (ResNet-50)': {
        'file': 'resnet50_scene.pth',
        'params': '23.5M',
        'speed': '5.72 ms',
        'input_size': 224,
        'tasks': ['cls'],
        'description': 'ResNet-50 trained independently on VisDrone-2019 for scene classification. Serves as the single-task baseline.'
    },
    'Single-Task Segmentation (ResNet-50)': {
        'file': 'resnet50_seg.pth',
        'params': '31.5M',
        'speed': '7.39 ms',
        'input_size': 256,
        'tasks': ['seg'],
        'description': 'ResNet-50 backbone with a U-Net style decoder trained on UAVid for 8-class pixel-level semantic segmentation.'
    },
    'Standard MTL': {
        'file': 'mtl_standard.pth',
        'params': '24.6M',
        'speed': '5.47 ms',
        'input_size': 256,
        'tasks': ['seg','cls'],
        'description': 'Shared ResNet-50 encoder with separate segmentation and classification heads, trained jointly using uncertainty weighting.'
    },
    'Cross-Task Attention MTL': {
        'file': 'mtl_cross_attention.pth',
        'params': '25.7M',
        'speed': '6.43 ms',
        'input_size': 256,
        'tasks': ['seg','cls'],
        'description': 'Proposed multi-task model with a cross-task attention module that lets each head attend to the other task\'s features.'
    },
    '3-Task MTL (Seg + Cls + Det)': {
        'file': 'three_task_mtl_best.pth',
        'params': '26.3M',
        'speed': '6.85 ms',
        'input_size': 256,
        'tasks': ['seg','cls','det'],
        'description': 'Unified model that performs segmentation, scene classification, and anchor-free object detection simultaneously.'
    }
}


class MultiTaskModel(nn.Module):
    def __init__(self, num_seg_classes=8, num_scene_classes=5):
        super().__init__()
        backbone = tvm.resnet50(weights=None)
        self.encoder_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.seg_up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.seg_conv1 = nn.Sequential(nn.Conv2d(512,512,3,padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.seg_up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_conv2 = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.seg_up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_conv3 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.seg_up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_conv4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.seg_up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_final = nn.Conv2d(32, num_seg_classes, 1)

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_scene_classes)
        )
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x, task='both'):
        feats = self.encoder_layers(x)
        out = {}
        if task in ('seg','both'):
            s = self.seg_up1(feats); s = self.seg_conv1(s)
            s = self.seg_up2(s); s = self.seg_conv2(s)
            s = self.seg_up3(s); s = self.seg_conv3(s)
            s = self.seg_up4(s); s = self.seg_conv4(s)
            s = self.seg_up5(s)
            out['seg'] = self.seg_final(s)
        if task in ('cls','both'):
            c = self.cls_pool(feats).flatten(1)
            out['cls'] = self.cls_fc(c)
        return out


class CrossTaskAttention(nn.Module):
    def __init__(self, feature_dim=512, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, query_feat, context_feat):
        b,c,h,w = query_feat.shape
        q = query_feat.flatten(2).permute(0,2,1)
        k = context_feat.flatten(2).permute(0,2,1)
        attended,_ = self.attn(q, k, k)
        attended = self.norm(attended + q)
        attended = self.proj(attended)
        return attended.permute(0,2,1).reshape(b,c,h,w)


class MultiTaskCrossAttention(nn.Module):
    def __init__(self, num_seg_classes=8, num_scene_classes=5):
        super().__init__()
        backbone = tvm.resnet50(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.seg_reduce = nn.Sequential(nn.Conv2d(2048,512,1), nn.ReLU())
        self.cls_reduce = nn.Sequential(nn.Conv2d(2048,512,1), nn.ReLU())
        self.seg_attend_cls = CrossTaskAttention(512, num_heads=4)
        self.cls_attend_seg = CrossTaskAttention(512, num_heads=4)

        self.seg_up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.seg_c1 = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.seg_up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.seg_c2 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.seg_up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.seg_c3 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.seg_up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.seg_up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.seg_final = nn.Conv2d(16, num_seg_classes, 1)

        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_scene_classes)
        )
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x, task='both'):
        shared = self.encoder(x)
        seg_feat = self.seg_reduce(shared)
        cls_feat = self.cls_reduce(shared)
        out = {}
        if task in ('seg','both'):
            enh = self.seg_attend_cls(seg_feat, cls_feat)
            s = self.seg_up1(enh); s = self.seg_c1(s)
            s = self.seg_up2(s);   s = self.seg_c2(s)
            s = self.seg_up3(s);   s = self.seg_c3(s)
            s = self.seg_up4(s);   s = self.seg_up5(s)
            out['seg'] = self.seg_final(s)
        if task in ('cls','both'):
            enh = self.cls_attend_seg(cls_feat, seg_feat)
            c = self.cls_pool(enh).flatten(1)
            out['cls'] = self.cls_head(c)
        return out


class ThreeTaskMTL(nn.Module):
    def __init__(self, num_seg=8, num_cls=5):
        super().__init__()
        backbone = tvm.resnet50(weights=None)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.up1 = nn.ConvTranspose2d(2048, 512, 2, stride=2)
        self.c1 = nn.Sequential(nn.Conv2d(512,512,3,padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c2 = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_head = nn.Conv2d(32, num_seg, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_cls)
        )
        self.det_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 128, 3, padding=1),  nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, 1), nn.Sigmoid()
        )
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_det = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f = self.encoder(x)
        s = self.c1(self.up1(f))
        s = self.c2(self.up2(s))
        s = self.c3(self.up3(s))
        s = self.c4(self.up4(s))
        s = self.up5(s)
        seg = self.seg_head(s)
        cls = self.classifier(self.pool(f).flatten(1))
        det = self.det_head(f)
        return seg, cls, det


class SegModelStandalone(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.encoder = tvm.resnet50(weights=None)
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(nn.Conv2d(512,512,3,padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.ReLU(), nn.BatchNorm2d(256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(128,128,3,padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(64,64,3,padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        f = self.encoder.conv1(x)
        f = self.encoder.bn1(f)
        f = self.encoder.relu(f)
        f = self.encoder.maxpool(f)
        f = self.encoder.layer1(f)
        f = self.encoder.layer2(f)
        f = self.encoder.layer3(f)
        f = self.encoder.layer4(f)
        x = self.up1(f);  x = self.conv1(x)
        x = self.up2(x);  x = self.conv2(x)
        x = self.up3(x);  x = self.conv3(x)
        x = self.up4(x);  x = self.conv4(x)
        x = self.up5(x)
        x = self.final(x)
        return x


HF_REPO = 'arpitjoshua/uav-multitask-models'

def download_from_hf(filename, local_path):
    url = f'https://huggingface.co/{HF_REPO}/resolve/main/{filename}'
    try:
        with st.spinner(f'Downloading {filename} from Hugging Face (first-time only)...'):
            urllib.request.urlretrieve(url, local_path)
        return True
    except Exception as e:
        st.error(f'Download failed: {e}')
        return False


@st.cache_resource
def load_model(model_name):
    device = torch.device('cpu')
    weights_dir = 'models'
    os.makedirs(weights_dir, exist_ok=True)
    fname = MODEL_INFO[model_name]['file']
    fpath = os.path.join(weights_dir, fname)

    if not os.path.exists(fpath):
        ok = download_from_hf(fname, fpath)
        if not ok:
            return None, False

    if model_name == 'Single-Task Classifier (ResNet-50)':
        m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    elif model_name == 'Single-Task Segmentation (ResNet-50)':
        m = SegModelStandalone()
    elif model_name == 'Standard MTL':
        m = MultiTaskModel()
    elif model_name == 'Cross-Task Attention MTL':
        m = MultiTaskCrossAttention()
    elif model_name == '3-Task MTL (Seg + Cls + Det)':
        m = ThreeTaskMTL()
    else:
        return None, False

    try:
        state = torch.load(fpath, map_location=device)
        m.load_state_dict(state)
        loaded = True
    except Exception as e:
        st.warning(f'Could not load weights: {str(e)[:200]}')
        loaded = False

    m.eval()
    return m, loaded


def make_transform(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])


def preprocess_image(pil_img, model_name):
    size = MODEL_INFO[model_name]['input_size']
    img_resized = pil_img.resize((size, size))
    img_array = np.array(img_resized)
    tfm = make_transform(size)
    tensor = tfm(pil_img).unsqueeze(0)
    return tensor, img_array


def seg_to_rgb(seg_mask):
    h, w = seg_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(8):
        rgb[seg_mask == c] = SEG_COLORS[c]
    return rgb


def blend(base, overlay, alpha=0.5):
    return (base.astype(np.float32)*(1-alpha) + overlay.astype(np.float32)*alpha).astype(np.uint8)


def heatmap_to_rgb(heat):
    h = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    cmap = plt.get_cmap('jet')
    col = cmap(h)[:,:,:3]
    return (col * 255).astype(np.uint8)


def run_inference(model, model_name, img_tensor, img_resized):
    results = {}
    start = time.time()
    with torch.no_grad():
        if model_name == 'Single-Task Classifier (ResNet-50)':
            out = model(img_tensor)
            probs = F.softmax(out, dim=1)[0].numpy()
            results['cls_probs'] = probs
            results['cls_pred'] = int(out.argmax(1).item())

        elif model_name == 'Single-Task Segmentation (ResNet-50)':
            out = model(img_tensor)
            out = F.interpolate(out, size=img_resized.shape[:2], mode='bilinear', align_corners=False)
            results['seg_mask'] = out.argmax(1)[0].numpy()

        elif model_name in ('Standard MTL', 'Cross-Task Attention MTL'):
            out = model(img_tensor, task='both')
            seg_out = F.interpolate(out['seg'], size=img_resized.shape[:2], mode='bilinear', align_corners=False)
            results['seg_mask'] = seg_out.argmax(1)[0].numpy()
            probs = F.softmax(out['cls'], dim=1)[0].numpy()
            results['cls_probs'] = probs
            results['cls_pred'] = int(out['cls'].argmax(1).item())

        elif model_name == '3-Task MTL (Seg + Cls + Det)':
            seg_out, cls_out, det_out = model(img_tensor)
            seg_out = F.interpolate(seg_out, size=img_resized.shape[:2], mode='bilinear', align_corners=False)
            det_out = F.interpolate(det_out, size=img_resized.shape[:2], mode='bilinear', align_corners=False)
            results['seg_mask'] = seg_out.argmax(1)[0].numpy()
            probs = F.softmax(cls_out, dim=1)[0].numpy()
            results['cls_probs'] = probs
            results['cls_pred'] = int(cls_out.argmax(1).item())
            results['det_heatmap'] = det_out[0,0].numpy()

    results['inference_time_ms'] = (time.time() - start) * 1000
    return results


def plot_classification_bars(probs, pred, figsize=(4,2.8)):
    fig, ax = plt.subplots(figsize=figsize)
    cols = ['#667eea' if i == pred else '#d0d0d8' for i in range(len(probs))]
    ax.barh(SCENE_CLASSES, probs, color=cols, edgecolor='none')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    plt.tight_layout()
    return fig


def render_result_row(results, img_resized, model_name):
    blocks = []
    blocks.append(('Input', img_resized, None))

    if 'seg_mask' in results:
        seg_rgb = seg_to_rgb(results['seg_mask'])
        overlay = blend(img_resized, seg_rgb, alpha=0.5)
        blocks.append(('Segmentation', overlay, None))

    if 'det_heatmap' in results:
        hrgb = heatmap_to_rgb(results['det_heatmap'])
        overlay = blend(img_resized, hrgb, alpha=0.4)
        blocks.append(('Object Detection', overlay, None))

    if 'cls_probs' in results:
        blocks.append(('Scene Classification', None, 'cls'))

    cols = st.columns(len(blocks))
    for col, (title, img, kind) in zip(cols, blocks):
        with col:
            st.markdown(f"**{title}**")
            if img is not None:
                st.image(img, use_column_width=True)
            elif kind == 'cls':
                fig = plot_classification_bars(results['cls_probs'], results['cls_pred'])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                pred_name = SCENE_CLASSES[results['cls_pred']]
                conf = results['cls_probs'][results['cls_pred']] * 100
                st.markdown(f"**Prediction:** `{pred_name}` ({conf:.1f}%)")


def render_segmentation_legend():
    cols = st.columns(8)
    for i, (cls, color) in enumerate(zip(SEG_CLASSES, SEG_COLORS)):
        with cols[i]:
            hex_c = '#{:02x}{:02x}{:02x}'.format(*color)
            st.markdown(
                f"<div style='background:{hex_c}; height:18px; border-radius:3px; margin-bottom:2px;'></div>"
                f"<div style='font-size:11px; text-align:center;'>{cls}</div>",
                unsafe_allow_html=True
            )


st.markdown('<p class="main-header">🛰️ Attention-Guided Multi-Task UAV Image Interpretation</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unified deep learning for drone imagery — semantic segmentation · scene classification · object detection</p>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### Navigation")
    mode = st.radio("Mode", ['Single Model Inference', 'Model Comparison', 'About the Project'], label_visibility='collapsed')

    st.markdown("---")
    st.markdown("### Datasets")
    st.markdown("""
**UAVid v1** — 670 high-resolution aerial images with pixel-level annotations across 8 semantic classes.

**VisDrone-2019** — 7,019 drone images with object bounding boxes, used to derive 5 scene categories.

All models run at the resolution they were trained on.
""")

    st.markdown("---")
    st.markdown("### Author")
    st.markdown("""
**Arpit Joshua Elias**
Student ID: 24257567

MSc in Artificial Intelligence
National College of Ireland

Supervisor: Prof. Furqan Rustam
""")


if mode == 'Single Model Inference':
    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("### Model")
        model_name = st.selectbox(
            "Choose a model",
            list(MODEL_INFO.keys()),
            label_visibility='collapsed'
        )
        info = MODEL_INFO[model_name]
        st.caption(info['description'])

        st.markdown(f"""
<div class="metric-card">
<div class="metric-val">{info['params']}</div>
<div class="metric-lbl">Parameters</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
<div class="metric-val">{info['speed']}</div>
<div class="metric-lbl">Inference Speed (CPU)</div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="metric-card" style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);">
<div class="metric-val">{info['input_size']}×{info['input_size']}</div>
<div class="metric-lbl">Input Resolution</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Input")
        source = st.radio("Source", ['Sample image', 'Upload your own'], label_visibility='collapsed')

        pil_img = None
        if source == 'Upload your own':
            up = st.file_uploader("JPG or PNG", type=['jpg','jpeg','png'], label_visibility='collapsed')
            if up:
                pil_img = Image.open(up).convert('RGB')
        else:
            samples_dir = 'samples'
            if os.path.exists(samples_dir):
                files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
                if files:
                    sel = st.selectbox("Pick a sample", files, label_visibility='collapsed')
                    pil_img = Image.open(os.path.join(samples_dir, sel)).convert('RGB')
                else:
                    st.warning("No samples available")
            else:
                st.warning("Samples folder missing")

    with col_main:
        if pil_img is None:
            st.info("← Select a sample image or upload your own to run the model.")
            st.markdown("""
### What this app shows

This application is the deployment artefact for an MSc research project on multi-task learning for UAV imagery. It lets you try any of the trained models interactively and see all their outputs in a single view.

**Available models**
- A single-task classifier and a single-task segmentation baseline
- A standard multi-task model (shared encoder + independent heads)
- The proposed cross-task attention model, which allows task heads to exchange information
- A three-task variant that additionally performs anchor-free object detection

**How to use**
1. Pick a model from the dropdown on the left
2. Choose a sample image or upload one of your own
3. The input is pre-processed to the resolution each model was trained on, then run through the model on CPU

**Note on real-world images**
The segmentation models were trained only on UAVid. When you upload out-of-distribution aerial photos (different altitudes, angles, or cameras), the predictions will be noticeably less accurate. For the best results, try the provided samples first.
""")
        else:
            model, loaded = load_model(model_name)

            if model is None:
                st.error("Model could not be loaded. Please check your internet connection.")
            else:
                if not loaded:
                    st.warning("Weights could not be loaded. Showing architecture with random initialisation — predictions will not be meaningful.")

                img_tensor, img_resized = preprocess_image(pil_img, model_name)
                results = run_inference(model, model_name, img_tensor, img_resized)

                st.markdown(f"### Results — {model_name}")
                st.caption(f"Inference time: {results['inference_time_ms']:.1f} ms (CPU)")

                render_result_row(results, img_resized, model_name)

                if 'seg_mask' in results:
                    st.markdown("**Segmentation classes**")
                    render_segmentation_legend()

                    with st.expander("Pixel class distribution"):
                        unique, counts = np.unique(results['seg_mask'], return_counts=True)
                        total = results['seg_mask'].size
                        fig, ax = plt.subplots(figsize=(10, 3))
                        pcts, labs, cls = [], [], []
                        for u, c in zip(unique, counts):
                            pcts.append(c/total*100)
                            labs.append(SEG_CLASSES[u])
                            cls.append(SEG_COLORS[u]/255)
                        ax.barh(labs, pcts, color=cls, edgecolor='none')
                        ax.set_xlabel('Percentage of pixels')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)


elif mode == 'Model Comparison':
    st.markdown("### Compare two models side by side")
    st.caption("Run two models on the same input image and see the differences in their predictions.")

    c1, c2 = st.columns(2)
    with c1:
        model_a = st.selectbox("Model A", list(MODEL_INFO.keys()), index=2)
    with c2:
        model_b = st.selectbox("Model B", list(MODEL_INFO.keys()), index=3)

    st.markdown("---")

    src = st.radio("Image source", ['Sample image', 'Upload your own'], horizontal=True)
    pil_img = None
    if src == 'Upload your own':
        up = st.file_uploader("JPG or PNG", type=['jpg','jpeg','png'])
        if up:
            pil_img = Image.open(up).convert('RGB')
    else:
        samples_dir = 'samples'
        if os.path.exists(samples_dir):
            files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
            if files:
                sel = st.selectbox("Pick a sample", files)
                pil_img = Image.open(os.path.join(samples_dir, sel)).convert('RGB')

    if pil_img is not None:
        model_a_obj, loaded_a = load_model(model_a)
        model_b_obj, loaded_b = load_model(model_b)

        if model_a_obj is not None and model_b_obj is not None:
            tensor_a, resized_a = preprocess_image(pil_img, model_a)
            tensor_b, resized_b = preprocess_image(pil_img, model_b)
            results_a = run_inference(model_a_obj, model_a, tensor_a, resized_a)
            results_b = run_inference(model_b_obj, model_b, tensor_b, resized_b)

            st.markdown("### Results")

            col_a, col_b = st.columns(2)

            for col, model_n, results, resized, inf in [
                (col_a, model_a, results_a, resized_a, MODEL_INFO[model_a]),
                (col_b, model_b, results_b, resized_b, MODEL_INFO[model_b])
            ]:
                with col:
                    st.markdown(f"#### {model_n}")
                    st.caption(f"{inf['params']} params · {inf['speed']} · {results['inference_time_ms']:.1f} ms inference · {inf['input_size']}×{inf['input_size']}")

                    st.image(resized, caption='Input', use_column_width=True)

                    if 'seg_mask' in results:
                        seg_rgb = seg_to_rgb(results['seg_mask'])
                        overlay = blend(resized, seg_rgb, alpha=0.5)
                        st.image(overlay, caption='Segmentation overlay', use_column_width=True)

                    if 'det_heatmap' in results:
                        hrgb = heatmap_to_rgb(results['det_heatmap'])
                        overlay = blend(resized, hrgb, alpha=0.4)
                        st.image(overlay, caption='Object detection heatmap', use_column_width=True)

                    if 'cls_probs' in results:
                        fig = plot_classification_bars(results['cls_probs'], results['cls_pred'], figsize=(5, 2.5))
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        pred_name = SCENE_CLASSES[results['cls_pred']]
                        conf = results['cls_probs'][results['cls_pred']] * 100
                        st.markdown(f"**Prediction:** `{pred_name}` ({conf:.1f}%)")

            if any('seg_mask' in r for r in [results_a, results_b]):
                st.markdown("---")
                st.markdown("**Segmentation classes**")
                render_segmentation_legend()


else:
    st.markdown("## About this project")

    st.markdown("""
This project investigates whether a unified multi-task deep learning framework can outperform independently trained single-task models on UAV imagery, while using fewer computational resources. The central contribution is a **cross-task attention module** that lets the segmentation and classification heads share intermediate features during inference.

The work follows the CRISP-DM methodology and is evaluated on two public drone datasets: **UAVid** for semantic segmentation and **VisDrone-2019** for scene classification and object detection. All experiments are conducted on Kaggle with GPU T4 x2 hardware, and the final models are deployed here for interactive evaluation.
""")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Best segmentation (mIoU)", "35.83%", "Det + Seg pair")
    with c2:
        st.metric("Best classification (acc)", "62.2%", "Swin MTL")
    with c3:
        st.metric("Best detection (F1)", "78.1%", "Det + Seg pair")

    st.markdown("---")

    st.markdown("### Research question")
    st.info(
        "Can a multi-task deep learning architecture with cross-task attention achieve superior aggregate performance across semantic segmentation, object detection, and scene classification on UAV imagery, compared with standard multi-task learning, independently trained single-task baselines, and traditional ML models?"
    )

    st.markdown("### Key findings")
    st.markdown("""
1. **Cross-task attention significantly improves scene classification** (paired t-test across 3 seeds, p = 0.022) without a significant drop in segmentation quality (p = 0.304).
2. **Segmentation and detection are synergistic.** Training them jointly pushes segmentation from 34.9% (single-task) up to 35.8% mIoU.
3. **A single 3-task model handles all three tasks successfully**, with 31.8% mIoU, 62.6% accuracy, and 73.7% detection F1, while using 61.6% fewer FLOPs than running the three single-task models separately.
4. **Backbone choice matters.** ResNet-50 is consistently stronger than EfficientNet-B3 on this data, and the Swin Transformer variant gives the best classification accuracy overall.
5. **Boundary quality is the biggest weakness.** Per-class mIoU drops sharply for thin classes like pedestrians and small vehicles, and boundary IoU is roughly half the interior IoU — this is the most promising direction for future work.
""")

    st.markdown("### Ablation results")
    import pandas as pd
    df = pd.DataFrame({
        'Model': [
            'Seg only (ResNet-50)', 'Cls only (ResNet-50)',
            'Seg + Cls pair', 'Det + Seg pair', 'Det + Cls pair',
            'Standard MTL', 'Cross-Task Attention MTL',
            'Swin Transformer MTL', 'EfficientNet-B3 MTL',
            '3-Task MTL'
        ],
        'Seg mIoU': [0.3446, '—', 0.3182, 0.3583, '—', 0.3141, 0.2958, 0.3267, 0.2744, 0.3176],
        'Cls Acc':  ['—', 0.5712, 0.5839, '—', 0.5675, 0.5675, 0.5876, 0.6131, 0.5712, 0.5985],
        'Det F1':   ['—', '—', '—', 0.7814, 0.6954, '—', '—', '—', '—', 0.7367]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Technical stack")
    st.markdown("""
- **Deep learning**: PyTorch 2.x, torchvision
- **Architectures**: ResNet-50, EfficientNet-B3, Swin Transformer, Faster R-CNN
- **Training**: Kendall's uncertainty-weighted multi-task loss, two-phase warmup schedule
- **Evaluation**: per-class mIoU, mAP at 0.5 and 0.5:0.95, F1 score, confusion matrices, gradient cosine similarity across tasks, FLOPs and throughput analysis
- **Statistical validation**: paired t-tests across three random seeds
- **Deployment**: Streamlit on Streamlit Community Cloud, with model weights hosted on Hugging Face Hub
""")

    st.markdown("### A note on limitations")
    st.markdown("""
The segmentation model has only seen UAVid training data, so predictions on arbitrary aerial photographs from the internet will not match its reported test-set performance. This domain shift is a known limitation of aerial scene understanding, and is discussed in the report as a direction for future work — for example, by adding domain adaptation or by training on a broader mixture of UAV datasets.
""")

    st.markdown("---")
    st.caption("Project submitted as part of the MSc Artificial Intelligence programme at National College of Ireland, April 2026.")
