# Proje ve Metodoloji

Bu proje perakende **talep tahmini ve stok optimizasyonu** için uçtan uca bir makine
öğrenmesi sistemidir. Amaç, aşırı stok (bağlı sermaye, depolama maliyeti) ile stok-out
(kayıp satış) arasında dengeyi kurmaktır.

## Yöntem
1. **Veri hazırlama** — 5 mağaza × 20 ürün × 760 günlük dengeli panel (76.000 satır).
2. **Özellik mühendisliği** — takvim özellikleri (ay, haftanın günü, döngüsel kodlama)
   ve mağaza-ürün bazında **lag/rolling** (gecikmeli ve hareketli ortalama) talep
   özellikleri. Sızıntı (leakage) yaratan `Units Sold`/`Units Ordered` çıkarılır.
3. **Modelleme** — XGBoost, Optuna ile hiperparametre optimizasyonu; zaman-bazlı
   train/test ayrımı. En iyi model held-out testte **R² ≈ 0.95**.
4. **Doğrulama** — rolling-origin (kayan pencere) zaman-serisi çapraz doğrulaması.
5. **Stok kararları** — tahminler güvenlik stoğu, yeniden sipariş noktası, ABC/ABC-XYZ,
   EOQ ve newsvendor politikalarına dönüştürülür.
6. **Asistan** — Azure OpenAI function-calling ile doğal dil arayüzü; kavramsal sorular
   için RAG (bilgi tabanı erişimi).

## Sonuçlar
Model tabanlı stok politikası, hedef %95 servis seviyesini korurken ortalama envanteri
naive politikaya göre **~%28 azaltır**, çünkü tam talep dalgalanması yerine yalnızca
küçük tahmin hatasını tamponlaması yeterlidir.
