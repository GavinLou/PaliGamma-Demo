import streamlit as st
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
import torch
from PIL import Image
import requests
from io import BytesIO
import time

st.set_page_config(
    page_title="PaliGemma Vision QA",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 20px; }
    .stTitle { color: #1f77b4; }
    .result-box { 
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """載入 PaliGemma 模型和 Processor"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner("正在載入 PaliGemma 模型... 這可能需要 1-2 分鐘"):
        model_id = "google/PaliGemma-3b-mix-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        processor = PaliGemmaProcessor.from_pretrained(model_id)
    
    return model, processor, device


def inference(model, processor, image, input_text, device):
    """執行模型推理"""
    try:
        # 準備輸入
        inputs = processor(
            text=input_text,
            images=image,
            padding="longest",
            do_convert_rgb=True,
            return_tensors="pt"
        ).to(device)
        
        inputs = inputs.to(dtype=model.dtype)
        
        # 執行推理
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_length=496)
        inference_time = time.time() - start_time
        
        # 解碼結果
        result = processor.decode(output[0], skip_special_tokens=True)
        
        return result, inference_time
    
    except Exception as e:
        return f"錯誤: {str(e)}", None


def load_image_from_url(url):
  
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"無法從 URL 載入圖片: {str(e)}")
        return None


def main():
    st.title("PaliGemma Vision Question Answering")
    st.markdown("用 AI 詢問圖片中的任何問題 | Ask AI anything about images")
    

    with st.sidebar:
        st.header("設定")
        device_info = st.session_state.device
        st.info(f"使用設備: {device_info}")
        
        if st.button("載入模型", use_container_width=True):
            st.session_state.model, st.session_state.processor, st.session_state.device = load_model()
            st.success("模型已載入!")
        
        if st.session_state.model is None:
            st.warning("請先點擊上方按鈕載入模型")
    
    col1, col2 = st.columns([1, 1], gap="large")
    

    with col1:
        st.subheader("圖片來源")
        
        image_source = st.radio(
            "選擇圖片來源:",
            ["上傳本機檔案", "輸入圖片 URL", "拍攝照片"],
            label_visibility="collapsed"
        )
        
        image = None
        
        if image_source == "上傳本機檔案":
            uploaded_file = st.file_uploader(
                "選擇圖片檔案 (JPG, PNG, GIF, WebP)",
                type=["jpg", "jpeg", "png", "gif", "webp"]
            )
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
        
        elif image_source == "輸入圖片 URL":
            url = st.text_input(
                "貼上圖片 URL:",
                placeholder="https://example.com/image.jpg"
            )
            if url and st.button("📥 載入圖片", use_container_width=True):
                image = load_image_from_url(url)
        
        elif image_source == "📷 拍攝照片":
            picture = st.camera_input("點擊拍攝照片")
            if picture:
                image = Image.open(picture).convert("RGB")
        
        if image:
            st.image(image, use_column_width=True, caption="已選擇圖片")
    
    with col2:
        st.subheader("問題")
        
        input_text = st.text_area(
            "輸入你的問題:",
            placeholder="例如: 圖片中有幾個人?",
            height=100
        )

        st.markdown("---")
        
        if st.button("詢問 AI", use_container_width=True, type="primary"):
            if st.session_state.model is None:
                st.error("請先在左側載入模型!")
            elif image is None:
                st.error("請先選擇一張圖片!")
            elif not input_text.strip():
                st.error("請輸入問題!")
            else:
                with st.spinner("AI 正在思考中..."):
                    result, inference_time = inference(
                        st.session_state.model,
                        st.session_state.processor,
                        image,
                        input_text,
                        st.session_state.device
                    )

                st.markdown("結果")
                
                result_col1, result_col2 = st.columns([2, 1])
                
                with result_col1:
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>AI 回答:</h4>
                        <p style="font-size: 16px; color: #1f77b4;"><b>{result}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    if inference_time:
                        st.metric("推理時間", f"{inference_time:.2f}s")
                    st.metric("模型大小", "3B 參數")
   
                st.code(result, language="text")

if __name__ == "__main__":
    main()

   