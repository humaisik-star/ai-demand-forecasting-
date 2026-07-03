# EOQ — Ekonomik Sipariş Miktarı

EOQ (Economic Order Quantity), her siparişte **ne kadar** sipariş verileceğini belirleyen
klasik envanter modelidir. Sipariş verme maliyeti ile stok tutma maliyeti arasındaki
dengeyi kurarak toplam maliyeti minimize eder.

## Formül
EOQ = √(2 · D · S / H)

- **D**: yıllık talep (adet)
- **S**: sipariş başına sabit maliyet (ordering cost)
- **H**: birim başına yıllık stok tutma maliyeti (holding cost). Genelde `H = tutma oranı ×
  birim fiyat` (ör. fiyatın %20'si).

## Sezgi
- Talep (D) veya sipariş maliyeti (S) arttıkça EOQ **artar** — daha seyrek, büyük sipariş
  mantıklıdır.
- Stok tutma maliyeti (H) arttıkça EOQ **azalır** — az stok tutmak için sık, küçük sipariş.

## İş anlamı
EOQ, sipariş verme ve stok taşıma maliyetlerini dengeleyerek en ekonomik parti
büyüklüğünü verir. Çok küçük siparişler sipariş maliyetini, çok büyük siparişler tutma
maliyetini şişirir. EOQ bu iki maliyetin toplamının minimum olduğu noktadır.
