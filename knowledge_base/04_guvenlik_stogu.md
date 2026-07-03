# Güvenlik Stoğu (Safety Stock)

Güvenlik stoğu, talep ve tedarik belirsizliğine karşı tutulan **tampon stoktur**.
Amaç, tedarik süresi (lead time) boyunca beklenmedik talep artışlarında stok-out
yaşamamaktır.

## Z-skorlu formül
Güvenlik Stoğu = z × σ_L = z × σ × √(lead time)

- **z**: hedef servis seviyesine karşılık gelen standart normal z-skoru. Örneğin %95
  servis seviyesi için z ≈ 1.645, %90 için ≈ 1.28, %99 için ≈ 2.33.
- **σ**: günlük talebin standart sapması.
- **lead time**: tedarik süresi (gün). Tedarik süresi boyunca belirsizlik σ × √(lead time)
  ile büyür.

## Servis seviyesi
Servis seviyesi arttıkça z büyür, dolayısıyla güvenlik stoğu artar. %95 → %99'a çıkmak
güvenlik stoğunu ciddi artırır; bu maliyet ile stok-out riski arasında bir tercihtir.

## Tahmine dayalı güvenlik stoğu
Talep iyi tahmin ediliyorsa, güvenlik stoğu tam talep dalgalanmasını (σ ≈ 40) değil,
yalnızca küçük **tahmin hatasını** (σ ≈ 6) tamponlamak zorundadır. Bu yüzden model
tabanlı politika, aynı servis seviyesinde çok daha az envanterle çalışır (~%28 azalma).
