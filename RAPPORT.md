## Kader voor het uitvoeren van experimenten
- Ik wil elk netwerk (Dutch, Roberta, Combined) apart trainen met dezelfde data en seed om grip te krijgen op de resultaten.

## Startpunt — vijf experimenten
- 01 Baseline: middel alle chunks tot één vector; een klein neuraal netwerk voorspelt 32 labels.
- 02 Attention: leer welke chunks belangrijk zijn en weeg ze voordat je voorspelt. Handig als slechts delen van een document het signaal bevatten.
- 03 BiLSTM: lees chunks op volgorde met een bidirectionele RNN, dan classificeer. Vangt sequentie/structuur maar is gevoelig voor padding en heeft meestal meer epochs nodig.
- 04 Highway: behoud het meeste van het originele signaal en gebruik een poort om alleen nuttige veranderingen toe te voegen. Stabiele prestaties vroeg in training met minder fouten.
- 05 MoE: gebruik meerdere kleine expert-netwerken en een poort die de beste twee per document kiest. Kan specialiseren naar verschillende patronen maar heeft tuning en trainingstijd nodig.

Dutch-Baseline       0.195        0.783        0.674        0.015       
Roberta-Baseline     0.177        0.787        0.673        0.014       
Combined-Baseline    0.175        0.789        0.679        0.014       
Dutch-Attention      0.405        0.845        0.747        0.011       
Roberta-Attention    0.343        0.827        0.726        0.012       
Combined-Attention   0.336        0.821        0.720        0.013       
Dutch-BiLSTM         0.433        0.865        0.773        0.010       
Roberta-BiLSTM       0.351        0.841        0.754        0.011       
Combined-BiLSTM      0.260        0.800        0.717        0.014       
Dutch-Highway        0.429        0.845        0.749        0.011       
Roberta-Highway      0.415        0.845        0.747        0.011       
Combined-Highway     0.375        0.843        0.735        0.012       
Dutch-MoE            0.209        0.788        0.673        0.014       
Roberta-MoE          0.192        0.785        0.669        0.014       
Combined-MoE         0.242        0.791        0.688        0.015   

- Dutch embeddings presteren beter dan Roberta en Combined bij alle architecturen
- Bij de individuele klasse-data zien we dat de slechtst presterende klassen (19, 23, 24, 20, 18, 25, 31, 26, 21, 28, 10, 30, 13, 29 & 8) veel minder voorbeelden bevatten.
- En waarom presteert het gecombineerde model (afgeleid van beide juridische modellen) niet beter dan de losse modellen?

## Kunnen de klassen met veel voorbeelden de klassen met weinig voorbeelden overschaduwen?

- Hoe presteert het model als ik alleen train op de slechtst presterende klassen?
- Hoe presteert het model als ik alle klassen evenveel input geef?

## Experiment 06 — Zeldzame Klassen
- 06 Rare Classes: filter de dataset zodat alleen documenten met zeldzame klassen overblijven (15 van de 32), en train een BiLSTM alleen daarop — om te testen of de vele "makkelijke" voorbeelden de schaarse klassen verdringen. 

## Experiment 07 — Minimum Klassen
- 07 Balanced Classes: neem van elke klasse evenveel voorbeelden (het minimum), zodat alle 32 klassen gelijk vertegenwoordigd zijn — om te testen of de scheve verdeling het probleem is of gewoon te weinig data.

## Vergelijking BiLSTM Strategieën (Heatmap)
- Comparison Heatmap: vergelijk E03 (volledig), E06 (zeldzaam) en E07 (gebalanceerd) in een heatmap per embedding type.
- Conclusie: Combined + gebalanceerd (E07) leert 9 klassen waaronder zeldzame klasse 20 — meer dan E03 (11 frequente) of E06 (5 klassen).
- Trade-off: micro F1 daalt van 80% naar 54%, maar het model herkent nu klassen die voorheen genegeerd werden.

## Experiment 08 — PCA + Trigram Preprocessing
- 08 PCA + Trigram: pas PCA toe om de 768-dim embeddings te reduceren naar 256 dimensies, en voeg 100 karakter-trigram features toe.
- Hypothese: PCA verwijdert ruis terwijl het signaal behouden blijft; trigrams vangen morfologische patronen in Nederlandse juridische tekst (voorvoegsels, achtervoegsels) die transformers mogelijk missen.

## Experiment 09 — Advanced PCA + Trigram met Skip Connections
- 09 Advanced: combineer Dutch PCA (256) + RoBERTa PCA (256) + Trigrams (100) = 612 features, met een dieper netwerk met skip connections.
- Resultaat: Combined wint nu met Macro-F1 0.3862 en Micro-F1 0.8520 — beter dan Dutch (0.3776) en RoBERTa (0.3031).
- Waarom: multimodale features vullen elkaar aan, PCA verwijdert ruis, skip connections stabiliseren training.
- 14 klassen blijven ongedetecteerd door te weinig voorbeelden.

## Experiment 10 — Adaptive Oversampling

Klasse 17 heeft 6470 docs, klasse 19 maar 43 — het model negeert zeldzame klassen. Ik kopieer zeldzame documenten zodat elke klasse ~3000 samples heeft (bijv. klasse 19 × 70 = 3010).

**Resultaat:** Macro-F1 verdubbeld (0.38 → 0.64), 30/31 klassen gedetecteerd (was 17). Klasse 19 haalt nu F1=0.91.

## Experiment 11 — Hyperparameter Tuning

Met adaptive oversampling als basis testte ik twee tuning-methodes:
- **Grid Search:** 24 combinaties van hidden layers, dropout en learning rate
- **Random Search:** 15 trials met random parameters (incl. batch size, weight decay, batchnorm, LR scheduler)

**Resultaat:** Random Search wint met Macro-F1 = **0.7058** (+10% vs E10).
- Beste params: hidden=[128,128], dropout=0.21, lr=0.0003, batch_size=16, batchnorm=True, StepLR scheduler
- Conclusie: kleinere netwerken met batchnorm en LR decay presteren beter dan grote netwerken.

## Experiment 12 — Final Model (100 epochs)

Beste E11 configuratie getraind voor 100 epochs met early stopping (patience=15).

**Resultaat:** Macro-F1 = **0.7236**, Micro-F1 = 0.8999, 30/31 klassen.
- Early stop bij epoch 28
- Klasse 22 haalt perfecte F1=1.0
- Zwakke klassen: 5 (0.53), 8 (0.40), 21 (0.48), **23 (0.00)**, 24 (0.60), 26 (0.17), 30 (0.56)

## Experiment 13 — Aggressive Oversampling

Hypothese: meer samples voor zwakke klassen (target 6000 i.p.v. 3000).

**Resultaat:** MISLUKT — Macro-F1 daalde naar 0.7062, zwakke klassen werden SLECHTER:
- Klasse 8: 0.40 → 0.29 ↓
- Klasse 21: 0.48 → 0.40 ↓  
- Klasse 23: 0.00 → 0.00 (blijft kapot)

**Conclusie:** Oversampling is niet de oplossing. Klasse 23 co-occurreert sterk met dominante klassen 17 (31.5%) en 12 (24.1%) — deze "stelen" de voorspellingen.

## Experiment 14 — Focal Loss + Adaptive Thresholds

Hypothese: focal loss focust op moeilijke voorbeelden, lagere drempelwaarde voor "gestolen" klassen.

**Resultaat:** Klasse 23 eindelijk gedetecteerd! F1=0.125 met threshold 0.2.
- Macro F1 = 0.7044 (lager door trade-off)
- Focal loss helpt, maar thresholds zijn belangrijker

## Experiment 15 — Threshold Optimization ⭐

Beste aanpak: train model normaal, optimaliseer thresholds per klasse achteraf.

**Resultaat:** Macro F1 = **0.7510** (+4.7% vs standaard)
- Klasse 23: F1=0.18 met threshold 0.20 (eindelijk gedetecteerd!)
- Klasse 30: F1=0.73 met threshold 0.65 (nu ⭐ status!)
- Klasse 5: F1=0.60 met threshold 0.05
- Klasse 24: F1=0.67 met threshold 0.40

**Wat is threshold optimalisatie?**
Normaal voorspelt het model: "als score > 0.5, dan is het deze klasse." Maar sommige klassen (zoals 23) krijgen nooit een score boven 0.5 — het model geeft 0.3 terwijl het eigenlijk correct is. Oplossing: per klasse een andere drempel kiezen. Het model verandert niet, alleen de beslissingsgrens per klasse.

## Experiment 16 — Threshold Hyperparameter Tuning

Systematisch thresholds tunen met ensemble van 5 modellen (5 epochs elk).

**Resultaat:** Macro F1 = **0.7595** (+3.9% vs standaard)
- Ensemble van 5 snelle modellen stabiliseert voorspellingen
- Thresholds geoptimaliseerd van 0.05 tot 0.85 per klasse
- Klasse 24: F1=0.73 ⭐ (threshold 0.35)
- Klasse 30: F1=0.75 ⭐ (threshold 0.60)
- **31/31 klassen gedetecteerd!**

## Experiment 17 — Ultimate Model ⭐⭐

Combinatie van alle beste technieken:
- Architectuur: [128,128] met BatchNorm, dropout=0.215
- Optimizer: Adam lr=0.000301, StepLR scheduler
- Ensemble: 5 modellen (100 epochs, early stop patience=15)
- Thresholds: geoptimaliseerd per klasse

**Resultaat:** Macro F1 = **0.7661** — BESTE SCORE! 🎉
- Micro F1 = 0.9070
- 4 perfecte klassen (F1=1.0): 2, 19, 22, 25
- **31/31 klassen gedetecteerd**

**Zwakke klassen vergeleken met E12:**
| Klasse | E12 | E17 | Verbetering |
|--------|-----|-----|-------------|
| 5 | 0.53 | 0.60 | +0.07 |
| 8 | 0.40 | 0.37 | -0.03 |
| 21 | 0.48 | **0.58** | +0.10 |
| **23** | **0.00** | **0.20** | **+0.20** |
| 24 | 0.60 | 0.67 | +0.07 |
| 26 | 0.17 | 0.17 | = |
| 30 | 0.56 | **0.75** ⭐ | +0.19 |

**Conclusie:** De combinatie van ensemble learning + threshold optimalisatie lost het probleem van "gestolen" voorspellingen op. Klasse 23 (altijd F1=0 in E01-E13) is nu detecteerbaar!

