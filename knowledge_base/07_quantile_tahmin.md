# Quantile (Kantil) Tahmin ve Tahmin Aralıkları

Nokta tahmini (tek bir sayı) talebin belirsizliğini gizler. **Quantile tahmin**, talebin
farklı kantilleri için ayrı tahminler üreterek bir **tahmin aralığı** verir.

## P10 / P50 / P90
- **P50**: medyan tahmin (talebin %50 olasılıkla altında/üstünde kaldığı değer).
- **P90**: talebin ancak **%10** olasılıkla aşacağı seviye. Yüksek servis için stok
  hedefi olarak kullanılır.
- **P10**: alt sınır; talebin %90 olasılıkla üstünde olduğu değer.

P10–P90 aralığı bir **%80 tahmin aralığıdır**.

## Nasıl eğitilir
XGBoost gibi modeller **pinball (quantile) kayıp fonksiyonu** ile her kantil için ayrı
eğitilir. Pinball kaybı asimetriktir: P90 için eksik tahmin, fazla tahminden ~9 kat daha
ağır cezalandırılır.

## Neden önemli — güvenlik stoğu bağlantısı
Talep normal dağılmıyorsa (kuyruklu), z-skorlu güvenlik stoğu yanıltıcı olabilir.
Quantile tahmin **dağılım-serbesttir**: P90 doğrudan "talebi %90 olasılıkla karşılayan"
stok seviyesini verir. Bu, güvenlik stoğunu ve newsvendor kararlarını normal varsayımı
olmadan besler.

## Kapsama (coverage)
Aralığın kalitesi PICP ile ölçülür: gerçek değerlerin yüzde kaçı P10–P90 içinde kalıyor.
İdeal ~%80. Bu projede held-out kapsama ~%72.7.
