# 📋 Yapılacaklar Listesi (Roadmap)

Projenin mevcut durumu ve sıradaki adımlar. Tamamladıkça `[ ]` → `[x]` yap.

---

## ✅ Tamamlananlar

- [x] Veri analizi (EDA) + grafikler (`analyze_data.py`, `results/`)
- [x] Feature engineering: takvim + lag/rolling (`src/features.py`)
- [x] Model eğitimi: train/validation/test ayrımı + Optuna tuning (`train_model.py`)
- [x] En iyi model: **XGBoost, R² ≈ 0.95** (held-out test)
- [x] Tahmin scripti (`predict.py`)
- [x] Stok optimizasyonu: güvenlik stoğu, reorder point, ~%28 stok azaltımı (`stock.py`)
- [x] LLM asistanı: Azure OpenAI function-calling (`assistant.py`, `src/assistant_tools.py`) — **#3 tamam**
- [x] README'de gerçek metrik tabloları (forecast + stok + backtest) — **#4 tamam**
- [x] Backtesting: rolling-origin zaman serisi CV (`backtest.py`) — **#2 tamam**
- [x] Test suite: 24 test, pytest (`tests/`)
- [x] CI: GitHub Actions ile her push'ta otomatik test (`.github/workflows/ci.yml`)
- [x] Azure ML scaffolding hazır (`azureml/`) — çalıştırmak için sadece abonelik gerekiyor

---

## 🔜 Sıradaki: Git & CI'yi aktive et

- [ ] Tüm çalışmayı commit'le
- [ ] GitHub'a push et → **CI otomatik çalışır** (Actions sekmesinde yeşil tik görürsün)
- [ ] README'deki rozetleri kontrol et (isteğe bağlı: CI badge ekle)

---

## ☁️ Azure ML'e Bağlanma (bulutta eğitim)

> **"Azure'a ne zaman bağlanacağız?"** → Ne zaman bir **Azure aboneliğin (subscription)**
> hazır olursa. Kod tarafında hazırız; aşağıdaki adımlar sadece Azure hesabı +
> `az` CLI kurulumu gerektiriyor. Yerelde her şey çalıştığı için acele yok.

> **"Azure'a bağlanınca eğitmeye devam edebilir miyim?"** → **Evet, kesinlikle.**
> Aynı `train_model.py` bulutta, Azure'ın compute cluster'ında çalışır. Farkı:
> daha güçlü makine, otomatik zamanlama (her gece yeniden eğitim), model versiyonlama
> ve REST API ile servis. Kodu değiştirmene gerek yok — sadece bir "job" tanımı yazılıyor.

### Kurulum
- [ ] Azure aboneliği aç (öğrenci isen [Azure for Students](https://azure.microsoft.com/free/students/) ücretsiz kredi verir)
- [ ] Azure CLI kur: `brew install azure-cli`, sonra `az login`
- [ ] ML uzantısı: `az extension add -n ml`
- [ ] Azure ML **Workspace** oluştur (portal veya CLI)

### Veri & Eğitim
> Kod hazır: `azureml/train-job.yml`, `azureml/conda.yml` ve adım adım komutlar
> `azureml/README.md` içinde. Aşağıdakiler sadece Azure hesabı gerektiriyor.
- [ ] `az login` + workspace oluştur (bkz. `azureml/README.md`)
- [ ] Compute cluster oluştur (`cpu-cluster`, scale-to-zero)
- [x] `train_model.py` için **job.yml** hazır (`azureml/train-job.yml`)
- [ ] `az ml job create -f azureml/train-job.yml --web` ile bulutta eğit
- [ ] Modeli **Model Registry**'ye kaydet (`az ml model create ...`)

### Servis & Otomasyon
- [ ] **Managed Online Endpoint** (gerçek zamanlı) veya **Batch Endpoint** (günlük toplu forecast) deploy et
- [ ] **Azure ML Pipeline** + schedule: her gece veri çek → forecast → stok öner
- [ ] (İsteğe bağlı) Model performansı düşünce otomatik yeniden eğitim (retraining trigger)

---

## 🤖 Azure OpenAI (LLM asistanı canlıya)

- [ ] Azure OpenAI / AI Foundry kaynağı oluştur
- [ ] Bir chat modeli deploy et (örn. `gpt-4o`)
- [ ] `.env.example` → `.env` kopyala, endpoint + key + deployment adını gir
- [ ] `python assistant.py --ask "hangi üründe stok riski var?"` ile test et
- [ ] (İleri seviye) Asistanı bir web arayüzüne bağla (Streamlit / FastAPI → Azure Container Apps)

---

## 🚀 Modeli Güçlendirme (isteğe bağlı ama etkili)

- [x] **Tahmin aralıkları** (quantile regression P10/P50/P90) — `quantile_forecast.py`
- [ ] **Çok-günlük forecast** (lead time boyunca her günü ayrı tahmin et)
- [x] **Backtesting** (`TimeSeriesSplit` ile kayan pencere doğrulaması) — `backtest.py`
- [x] **SHAP** ile açıklanabilirlik ("bu ürünün talebi neden arttı?") — `explain.py`
- [ ] **MLflow** ile deney takibi
- [ ] Dashboard'a (`dashboard.html`) forecast + stok sonuçlarını ekle

---

## 📌 Notlar
- Yerel çalıştırma sırası: `train_model.py` → `predict.py` → `stock.py` → `assistant.py`
- Testler: `python -m pytest`
- Azure'a geçiş kodu bozmaz; `src/features.py` paylaşımlı olduğu için her yerde aynı feature'lar üretilir.

## Envanter analitigi (ABC/EOQ/uyarilar) - TAMAM
- [x] ABC analizi (Pareto, A/B/C siniflari)
- [x] EOQ + reorder point
- [x] Stok tukenme uyarilari (CRITICAL/REORDER/OK)
- [x] Chatbot tool: abc_summary, get_inventory_policy, list_stockout_alerts

## Ileri seviye (ABC-XYZ / Newsvendor / Yonetici Ozeti) - TAMAM
- [x] ABC-XYZ matrisi (deger x talep degiskenligi)
- [x] Newsvendor optimal siparis (critical ratio)
- [x] Z-skorlu guvenlik stogu
- [x] Stok devir hizi + gun cinsinden stok
- [x] Anomali tespiti + aciklama
- [x] Otomatik Yonetici Ozeti (Azure OpenAI) - executive_summary.py + yonetici_ozeti tool
- [x] Raporlar sekmesinde ABC-XYZ matrisi + anomaliler

## RAG katmani (programin cekirdegi) - TAMAM
- [x] Bilgi tabani: 9 kavram dokumani (ABC, EOQ, newsvendor, guvenlik stogu, quantile...)
- [x] Chunking + Azure OpenAI embeddings (text-embedding-3-small) -> SQLite
- [x] Retrieval: cosine similarity, top-K (src/rag.py)
- [x] bilgi_ara tool (hibrit) - kaynak gostererek cevap, yoksa bilmiyorum
- [x] 10 soruluk eval: %100 dogruluk (rag_eval.py)
- [x] CI-safe testler (test_rag.py)

## SQLite veri katmani - TAMAM
- [x] build_db.py: sonuc CSV'lerini db/analysis.db'ye yaz
- [x] src/datastore.py + webapp/backend/store.py: DB'den oku, yoksa CSV fallback
- [x] Dashboard API + chatbot tool'lari DB'den okuyor
- [x] Mevcut CSV akisi korundu (ek katman)
