# Yeniden Sipariş Noktası (Reorder Point)

Yeniden sipariş noktası (ROP), stok seviyesinin **ne zaman** yeni sipariş verilmesi
gerektiğini gösteren eşiktir. Mevcut stok bu seviyenin altına düştüğünde sipariş verilir.

## Formül
ROP = (ortalama günlük talep × tedarik süresi) + güvenlik stoğu

- İlk terim: tedarik süresi (lead time) boyunca beklenen talep.
- İkinci terim: belirsizliğe karşı güvenlik stoğu.

## Sezgi
Sipariş verildikten sonra mal gelene kadar (lead time) talep devam eder. ROP, bu süre
boyunca tükenmemek için gereken stoğu artı güvenlik tamponunu kapsar. Tedarik süresi
uzadıkça veya talep arttıkça ROP yükselir.

## Stok durumu ve uyarılar
- Mevcut stok < güvenlik stoğu → **KRİTİK** (stok-out riski çok yüksek).
- Mevcut stok ≤ ROP → **REORDER** (sipariş zamanı).
- Aksi halde → **OK**.

## Gün cinsinden stok (days of cover)
Mevcut stok ÷ ortalama günlük talep = stoğun kaç gün yeteceği. Düşük değer stok-out
riski, çok yüksek değer fazla stok (bağlı sermaye) anlamına gelir.
