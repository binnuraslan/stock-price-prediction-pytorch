# PyTorch ile Hisse Senedi Fiyat Tahmini (LSTM vs GRU)

*Other languages: [English](README.md)*

Amazon (AMZN) hisse senedinin kapanış fiyatını, PyTorch ile geliştirilen iki farklı tekrarlayan sinir ağı (RNN) mimarisi — **LSTM** ve **GRU** — kullanarak tahmin eden ve performanslarını karşılaştıran bir makine öğrenmesi projesi.

Proje, Rodolfo Saldanha'nın [Stock price prediction with PyTorch](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632) başlıklı makalesindeki genel yaklaşımı temel alıyor.

## Genel Bakış

- **Görev**: zaman serisi regresyonu — önceki N günün verisiyle bir sonraki günün kapanış fiyatını tahmin etmek
- **Veri**: AMZN günlük kapanış fiyatı, ~14 yıllık veri (2012–günümüz), [yfinance](https://pypi.org/project/yfinance/) ile çekildi
- **Modeller**: 2 katmanlı bir LSTM ve 2 katmanlı bir GRU, aynı koşullar altında eğitildi ve değerlendirildi
- **Metrik**: test setinde MSE / RMSE

## Yöntem

1. **Veri toplama**: AMZN'nin günlük fiyat verisi `yfinance` ile indirildi; 2022'deki hisse bölünmesini (stock split) düzeltmek için `auto_adjust=True` kullanıldı.
2. **Ön işleme**:
   - Sadece `Close` (kapanış) fiyatı kullanıldı.
   - Eksik satırlar veriden çıkarıldı (`dropna`).
   - Fiyatlar `MinMaxScaler` ile `[-1, 1]` aralığına ölçeklendi.
3. **Kayan pencere**: bir sonraki günü tahmin etmek için **20 günlük** bir geçmiş pencere kullanıldı (özel bir `create_sequences` fonksiyonuyla diziler oluşturuldu).
4. **Train/test bölme**: %80 / %20, **kronolojik sırayla** (karıştırılmadan) bölündü — böylece model yalnızca kendi eğitim döneminden sonraki veriyle test edilmiş oldu, geleceğe dair bir bilgi sızıntısı (data leakage) önlendi.
5. **Modeller**: iki model de aynı hiperparametreleri paylaşıyor — `input_dim=1`, `hidden_dim=32`, `num_layers=2`, `output_dim=1` — böylece karşılaştırma, model kapasitesinden değil doğrudan mimari farktan (LSTM vs GRU) kaynaklanıyor.
6. **Eğitim**: MSE kayıp fonksiyonu, Adam optimizer (`lr=0.01`). LSTM 300 epoch, GRU ise 100 epoch eğitildi — LSTM'in benzer bir kayıp seviyesine ulaşması daha fazla epoch gerektirdi; bunun sebebi muhtemelen fazladan bir kapıya (ve dolayısıyla daha fazla öğrenilecek parametreye) sahip olması.

## Sonuçlar

| Model | Test MSE | Test RMSE | Eğitim süresi |
|-------|---------:|----------:|---------------:|
| LSTM  | 165.72   | 12.87     | — |
| GRU   | 100.26   | 10.01     | ~5.5 sn |

Bu veri setinde **GRU, LSTM'den daha iyi performans gösterdi** — hem doğrulukta hem eğitim hızında. Bu sonuç, referans makaledeki bulgularla tutarlı. Bunun olası sebebi, GRU'nun daha basit kapı yapısının (ayrı bir hücre durumu ve çıkış kapısı olmadan) daha az parametre içermesi ve bu sayede benzer miktarda eğitimle daha hızlı yakınsaması.

Her iki model de hissenin genel trendini makul ölçüde takip edebiliyor, ancak — hisse senedi tahmininde beklendiği gibi — ani ve keskin fiyat hareketlerini önceden öngörmek yerine bir adım geriden takip ediyorlar.

## Proje Yapısı

```
.
├── stockpricepredictionwithPyTorch.ipynb   # ana notebook (veri, modeller, eğitim, değerlendirme)
├── README.md
├── README.tr.md
└── requirements.txt
```

## Nasıl Çalıştırılır

1. Bu repository'yi klonla.
2. Bir sanal ortam oluştur ve bağımlılıkları kur:
   ```bash
   python3 -m venv ml_env
   source ml_env/bin/activate   # Windows: ml_env\Scripts\activate
   pip3 install -r requirements.txt
   ```
3. Jupyter'i başlat ve notebook'u baştan sona çalıştır:
   ```bash
   jupyter notebook
   ```
4. `stockpricepredictionwithPyTorch.ipynb`'yi aç, sonra **Kernel → Restart Kernel and Run All Cells** yap.

Rastgele bir tohum (seed) sabitlenmediği sürece sonuçlar her çalıştırmada hafifçe değişebilir — bu notebook'ta her model oluşturulmadan önce `torch.manual_seed(42)` ile bu sabitlenmiştir.

## Sınırlılıklar ve Değerlendirme

Sadece geçmiş fiyatlara dayanarak hisse senedi fiyatı tahmin etmek gerçekten zor bir problem — gerçek piyasalar, bu veri setinde hiç yer almayan sayısız faktörden (haberler, makroekonomi, yatırımcı psikolojisi) etkileniyor. Buradaki nispeten düşük RMSE, modelin geleceği öngörebildiğinden çok, yakın geçmişteki trendi takip etmeyi öğrendiğini gösteriyor. Bu proje öğrenme amaçlıdır (zaman serisi ön işleme, RNN'ler ve PyTorch eğitim döngüleri üzerine), bir yatırım/alım-satım aracı olarak tasarlanmamıştır.

## Teşekkür

- Yöntem referansı: [Rodolfo Saldanha — "Stock price prediction with PyTorch"](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)
- Veri: [Yahoo Finance](https://finance.yahoo.com/), `yfinance` Python paketi aracılığıyla
