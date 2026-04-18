# 🚀 نشر التطبيق على Streamlit Cloud

## خطوات النشر:

### 1️⃣ إنشاء حساب GitHub
- اذهب إلى [github.com](https://github.com)
- إنشاء حساب مجاني

### 2️⃣ رفع الملفات على GitHub
```bash
git init
git add .
git commit -m "Initial commit: Autism Prediction App"
git branch -M main
git remote add origin https://github.com/yourusername/autism-prediction.git
git push -u origin main
```

### 3️⃣ إنشاء حساب Streamlit Cloud
- اذهب إلى [streamlit.io/cloud](https://streamlit.io/cloud)
- انقر على "Sign up"
- استخدم حساب GitHub الخاص بك

### 4️⃣ نشر التطبيق
1. اضغط على "New app"
2. اختر المستودع (Repository)
3. اختر الفرع (Branch): `main`
4. اختر الملف: `app.py`
5. اضغط "Deploy"

---

## ✅ الملفات المطلوبة:

```
✅ app.py
✅ Autism_data.csv
✅ random_forest_model.pkl
✅ requirements.txt
✅ .gitignore
✅ .streamlit/config.toml
```

---

## ⚠️ المشاكل الشائعة:

### مشكلة: "FileNotFoundError: Autism_data.csv"
**الحل:** تأكد من وجود الملف في نفس مجلد `app.py`

### مشكلة: "ModuleNotFoundError: sklearn"
**الحل:** تأكد من `requirements.txt` يحتوي على `scikit-learn`

### مشكلة: التطبيق بطيء
**الحل:** الحد الأقصى للذاكرة على Streamlit Cloud هو 1GB
- حاول تقليل حجم البيانات
- استخدم `@st.cache_resource` (بالفعل موجود)

---

## 📊 الرابط النهائي

بعد النشر الناجح:
```
https://your-username-autism-prediction.streamlit.app
```

---

## 🔐 الأمان

**تحذير:** لا تضع كلمات مرور أو مفاتيح سرية في الكود!

استخدم Streamlit Secrets:
1. في لوحة التحكم → Settings
2. اضغط على "Secrets"
3. أضف بيانات حساسة هناك

```python
# في الكود:
api_key = st.secrets["api_key"]
```

---

## 📞 الدعم

- مستندات Streamlit: https://docs.streamlit.io
- المنتدى: https://discuss.streamlit.io
- GitHub Issues: https://github.com/streamlit/streamlit/issues
