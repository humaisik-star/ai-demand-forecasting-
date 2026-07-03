# Newsvendor (Gazete Bayii) Modeli

Newsvendor modeli, **tek dönemlik** (bozulabilir veya kısa raf ömürlü) ürünler için
optimal sipariş miktarını belirler. Klasik örnek: bir gazete bayii, satılmayanın değersiz
kaldığı bir ürün için o gün kaç adet sipariş vermeli?

## Kritik oran (critical ratio)
Optimal sipariş, talep dağılımının **kritik orana** karşılık gelen kantilinde tutulur:

Kritik oran = Cu / (Cu + Co)

- **Cu** (underage / az sipariş maliyeti): bir birim eksik sipariş edilince kaybedilen kâr
  (genelde `marj × fiyat`).
- **Co** (overage / fazla sipariş maliyeti): satılmayan bir birimin maliyeti (tutma /
  değer kaybı).

## Optimal sipariş
Q* = talep dağılımının F⁻¹(kritik oran) noktası. Normal varsayım altında:
Q* = μ + z(kritik oran) × σ (lead time üzerinden).

## Sezgi
- Az sipariş maliyeti (Cu) yüksekse (yüksek marj) → kritik oran büyür → **daha fazla**
  sipariş (stok-out pahalı).
- Fazla sipariş maliyeti (Co) yüksekse → kritik oran küçülür → **daha az** sipariş.

## Kantil tahminle bağlantı
Kritik oran doğrudan bir servis seviyesi/kantildir; bu yüzden **quantile tahmin** (P90
gibi) çıktıları newsvendor kararını dağılım-serbest biçimde besler.
