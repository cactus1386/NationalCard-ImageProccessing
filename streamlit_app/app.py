import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
import os
from utils import load_models, rotation, process_img

# Page configuration
st.set_page_config(
    page_title="استخراج اطلاعات کارت ملی",
    page_icon="🆔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.upload-section {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 2rem;
    text-align: center;
    background-color: #f8f9ff;
}
</style>
""", unsafe_allow_html=True)

def format_extracted_data(data):
    """Format extracted data with Persian labels"""
    field_mapping = {
        'National': 'کد ملی',
        'Name': 'نام',
        'LastName': 'نام خانوادگی', 
        'FatherName': 'نام پدر',
        'Birth': 'تاریخ تولد',
        'Expire': 'تاریخ انقضا'
    }
    
    formatted = {}
    for key, value in data.items():
        persian_key = field_mapping.get(key, key)
        formatted[persian_key] = value if value else "یافت نشد"
    
    return formatted

# Header
st.markdown("""
<div class="main-header">
    <h1>🆔 سیستم استخراج اطلاعات کارت ملی ایرانی</h1>
    <p>با استفاده از هوش مصنوعی و تکنولوژی‌های YOLO، MediaPipe و Hezar</p>
</div>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_ai_models():
    """Load all AI models with caching"""
    with st.spinner("⏳ در حال بارگذاری مدل‌های هوش مصنوعی..."):
        try:
            models = load_models()
            st.success("✅ مدل‌ها با موفقیت بارگذاری شدند!")
            return models
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری مدل‌ها: {str(e)}")
            return None

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 آپلود تصویر کارت ملی")
    
    uploaded_file = st.file_uploader(
        "فایل تصویر را انتخاب کنید",
        type=['jpg', 'jpeg', 'png', 'gif'],
        help="فرمت‌های پشتیبانی شده: JPG, PNG, GIF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="تصویر آپلود شده", use_column_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file is not None:
        st.subheader("🚀 پردازش تصویر")
        
        if st.button("▶️ شروع پردازش", type="primary", use_container_width=True):
            try:
                # Load models
                models = load_ai_models()
                if models is None:
                    st.stop()
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Rotation and card detection
                status_text.text("🔄 در حال تشخیص و چرخش کارت...")
                progress_bar.progress(25)
                
                rotated_card = rotation(temp_path, models)
                
                if rotated_card is None:
                    st.error("❌ کارت ملی در تصویر شناسایی نشد!")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    st.stop()
                
                # Step 2: Text extraction
                status_text.text("📝 در حال استخراج متن...")
                progress_bar.progress(75)
                
                extracted_data = process_img(rotated_card, models)
                
                # Step 3: Complete
                status_text.text("✅ پردازش کامل شد!")
                progress_bar.progress(100)
                
                # Display processed card
                st.subheader("🆔 کارت پردازش شده")
                st.image(rotated_card, caption="کارت تصحیح شده", use_column_width=True)
                
                # Display extracted data
                st.subheader("📊 اطلاعات استخراج شده")
                
                if extracted_data:
                    formatted_data = format_extracted_data(extracted_data)
                    
                    # Create DataFrame more safely
                    try:
                        df_data = []
                        for key, value in formatted_data.items():
                            df_data.append({"فیلد": str(key), "مقدار": str(value)})
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Download results - Fixed CSV encoding
                        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 دانلود نتایج (CSV)",
                            data=csv_data.encode('utf-8-sig'),
                            file_name=f"extracted_data_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as csv_error:
                        st.error(f"خطا در ایجاد جدول: {csv_error}")
                        # Show data as JSON instead
                        st.json(formatted_data)
                    
                    # Technical details in expander
                    with st.expander("🔍 جزئیات فنی"):
                        st.json(extracted_data)
                        
                else:
                    st.warning("⚠️ هیچ متنی از تصویر استخراج نشد!")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"❌ خطا در پردازش: {str(e)}")
                # Show detailed error for debugging
                st.exception(e)
                
                # Clean up temp file in case of error
                try:
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass

# Statistics section
if uploaded_file is not None:
    st.markdown("---")
    st.subheader("📈 آمار پردازش")
    
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.metric("وضعیت", "آماده پردازش", "✅")
    
    with col_stats2:
        st.metric("فرمت فایل", uploaded_file.type.split('/')[-1].upper())
    
    with col_stats3:
        st.metric("حجم فایل", f"{uploaded_file.size / 1024:.1f} KB")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>ساخته شده با ❤️ | استفاده از YOLO, MediaPipe و Hezar</p>
</div>
""", unsafe_allow_html=True)
