# Stok Devir Hızı ve Talep Tahmini

## Stok devir hızı (inventory turnover)
Stok devir hızı, envanterin bir yılda kaç kez "döndüğünü" (satılıp yenilendiğini) ölçer:

Devir Hızı = Yıllık Talep / Ortalama Envanter

- **Yüksek devir**: stok hızlı satılıyor, sermaye verimli kullanılıyor, ama stok-out
  riski daha yüksek.
- **Düşük devir**: fazla stok, bağlı sermaye ve raf/depolama maliyeti.

**Gün cinsinden stok** = 365 / devir hızı. Ortalama bir birimin rafta kaç gün beklediğini
gösterir.

## Talep tahmini (demand forecasting)
Talep tahmini, geçmiş veriden gelecekteki talebi öngörme işidir. Bu projede günlük
mağaza-ürün talebi XGBoost ile tahmin edilir.

Güçlü tahminin anahtarı **özellik mühendisliğidir**:
- **Lag özellikleri**: dünkü (lag_1), geçen haftaki (lag_7) talep. Talep güçlü şekilde
  otokorelasyonlu olduğu için bunlar çok bilgilendiricidir.
- **Rolling (hareketli) özellikler**: son 7/30 günün ortalaması ve standart sapması.
- **Takvim özellikleri**: ay, haftanın günü, mevsim (döngüsel kodlanır).

Bu özellikler sayesinde model held-out testte **R² ≈ 0.95** doğruluğa ulaşır. Tahmin
doğruluğu; RMSE (ortalama karesel hata karekökü), MAE (ortalama mutlak hata) ve R²
(açıklanan varyans) ile ölçülür.
