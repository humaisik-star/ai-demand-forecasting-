# ABC-XYZ Analizi

ABC-XYZ, iki boyutu birleştiren bir segmentasyondur: **ABC** ürünün değerini (ciro),
**XYZ** ise talebin **değişkenliğini** (öngörülebilirliğini) ölçer. Sonuç 9 hücreli bir
matristir.

## XYZ — talep değişkenliği
Değişim katsayısına (CV = standart sapma / ortalama talep) göre:
- **X**: CV ≤ 0.5 — **stabil**, öngörülebilir talep.
- **Y**: 0.5 < CV ≤ 1.0 — **değişken** talep (trend/mevsim etkili).
- **Z**: CV > 1.0 — **erratik**, düzensiz talep.

## 9 hücreli matris ve strateji
- **AX** (yüksek değer, stabil): otomasyona en uygun; sıkı ama basit kontrol, düşük
  güvenlik stoğu yeterli.
- **AZ** (yüksek değer, erratik): en zor segment; yüksek güvenlik stoğu ve yakın takip.
- **CX** (düşük değer, stabil): basit, otomatik yeniden sipariş.
- **CZ** (düşük değer, erratik): minimum stok, gerekirse siparişe göre üretim/tedarik.

## İş anlamı
Sadece ciro (ABC) yeterli değildir; aynı A sınıfında stabil (AX) ve erratik (AZ) ürünler
çok farklı stok politikaları gerektirir. XYZ boyutu, güvenlik stoğu ve tahmin çabasının
nereye yoğunlaşacağını belirler.
