# ABC Analizi

ABC analizi, ürünleri (SKU) yıllık ciro katkısına göre önem sırasına dizen bir
segmentasyon yöntemidir. **Pareto ilkesine** (80/20) dayanır: ürünlerin küçük bir
kısmı cironun büyük kısmını üretir.

## Sınıflandırma
Ürünler yıllık ciroya göre büyükten küçüğe sıralanır ve **kümülatif ciro payına** göre
etiketlenir:
- **A sınıfı**: kümülatif cironun ilk **%80**'ini oluşturan ürünler — az sayıda ama en
  değerli. Yüksek servis seviyesi, sık takip, otomatik yeniden sipariş.
- **B sınıfı**: sonraki **%80–95** dilimi — orta önem, aylık gözden geçirme.
- **C sınıfı**: son **%95–100** dilimi — çok sayıda ama düşük değerli; minimum stok,
  basit kontrol.

## İş anlamı
Kaynaklar (dikkat, güvenlik stoğu, sayım sıklığı) paranın olduğu yere yönlendirilir.
A sınıfı ürünlerde stok-out doğrudan gelir kaybıdır, bu yüzden önceliklidir. C sınıfında
fazla stok tutmak sermaye israfıdır.

Bu projede yıllık ciro = ortalama günlük talep × 365 × birim fiyat ile hesaplanır.
Örnek dağılım: A = 59 SKU (cironun %79.8'i), B = 24, C = 17.
