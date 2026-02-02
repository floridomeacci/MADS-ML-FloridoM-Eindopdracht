Florido Meacci – 02/02/2026 
https://github.com/floridomeacci/MADS-ML-FloridoM 

Grip op zeldzame juridische klassen 

Toen ik de opdracht starten vroeg ik mij al snel af waarom er bepaalde juridische klassen structureel genegeerd werden door het model, ondankds dat de architectuur steeds beter wordt? 

Ik besloot dit onderzoek strikt gecontroleerd aan te pakken. Elk netwerk, Dutch, RoBERTa en een gecombineerd embedding-model, trainde ik afzonderlijk, met exact dezelfde data en random seed. In de hoop zo beter te omvatten waar de verschillen vandaan kwamen. 

Eerste hypothese: architectuur is de bottleneck 

Mijn startpunt bestond uit de vijf architecturen uit de gtihub, oplopend in complexiteit: een eenvoudige baseline (gemiddelde van chunks), attention, BiLSTM, highway-netwerken en een Mixture-of-Experts. 

Complexere architecturen zoals BiLSTM en attention leverden duidelijke winst op ten opzichte van de baseline, vooral in Macro-F1. Het gecombineerde model presteerde consistent beter dan zowel Dutch als RoBERTa afzonderlijk. Door beide embedding-types te combineren presteerd het model beter. 

Toen ik inzoomde op de individuele klasse prestaties, viel iets anders op. De slechtst presterende klassen waren de klassen met de minste voorbeelden. Velen werden simpelweg nooit voorspeld omdat er verreweg minder data voor was. 

 
 
 

 
 
 
 
Tweede hypothese: frequente klassen verdringen zeldzame klassen 

Hier begon het echte probleem zichtbaar te worden. Goed presterende klassen hadden duizenden voorbeelden, terwijl de laag preseterende klasses er minder hadden. Mijn vermoeden was dat het model leerde voorspellen wat statistisch het meest voorkomt. Om dit te testen draaide ik het probleem om. In Experiment 06 trainde ik een BiLSTM uitsluitend op documenten die zeldzame klassen bevatten. Het resultaat was nog steeds slecht: het model herkende nu slechts een handvol klassen. Zeldzame klassen alleen trainen bleek niet genoeg, er was simpelweg te weinig data. 

In Experiment 07 ging ik een stap verder en forceerde ik balans: van elke klasse maakte ik evenveel voorbeelden. De Micro-F1 kelderde, maar iets veranderde. Het gecombineerde embedding-model begon nu zeldzame klassen te leren die het voorheen volledig negeerde. Dit was een keerpunt. Het probleem was niet alleen dat sommige klassen zeldzaam waren, het probleem was dat ze overschaduwd werden. 

Derde hypothese: representatie is te ruisachtig 

Tot nu toe had ik alleen aan de buitenkant van het probleem gesleuteld: data en architectuur. In Experiment 08 keek ik naar de embeddings zelf. Ik reduceerde de 768-dimensionale vectoren met de PCA trigram techniek. De gedachte hierachter was dat PCA ruis zou verwijderen en trigrams patronen zou vangen die transformers mogelijk missen. Dat werkte, maar pas echt in Experiment 09, waar ik Dutch PCA, RoBERTa PCA en trigrams combineerde in één netwerk met skip connections. Voor het eerst won het gecombineerde model overtuigend. 

Mijn conclusie hier was duidelijk: gecombineerde embeddings werken vooral als je ze dwingt elkaar aan te vullen met skip layers in plaats van elkaar te verdringen. 

Vierde hypothese: het model ziet zeldzame klassen simpelweg te weinig 

In Experiment 10 ging ik nog een stapje verder. Ik oversamplede zeldzame klassen tot ongeveer 3000 voorbeelden per klasse. Het effect werkte. De Macro-F1 steeg van 0.16 naar 0.64, een verviervoudiging. Klassen die tot nu toe “onzichtbaar” waren, haalden ineens F1-scores boven de 0.9. Voor het eerst werden vrijwel alle klassen gedetecteerd.In Experiment 11 deed ik daar bovenop: hyperparameter tuning (grid search, random search en Bayesian optimalisatie) om de beste modelconfiguratie te vinden. Dit leverde een Macro-F1 van 0.70 op. 

In Experiment 12 paste ik deze beste hyperparameters toe. De Macro-F1 kwam uit op 0.69. In Experiment 13 probeerde ik agressiever te oversamplen (6000 voorbeelden voor struggling klassen). Dit leverde verdere verbetering op naar 0.72 en detecteerde nu 30 klassen. 

Maar waarom bleef klasse 23 hardnekkig steeds op F1=0 steken? Ik ging dieper in op deze klasse en kwam tot de conclusie dat oversampling alleen niet genoeg is, sommige klassen vereisen een andere beslissingsstrategie. In dit geval werd klasse 23 steeds overschaduwd door best presterende klasses in gepaarde data 

. 
 

 

Vijfde hypothese: het probleem zit niet in leren, maar in beslissen 

Tot dit punt had ik alles geprobeerd om het model beter te laten leren. Maar wat als het model het eigenlijk al weet, alleen niet durft te beslissen? Dat bracht me bij thresholds. 

In Experiment 14 paste ik focal loss toe en lagere trehsholds voor de zeldzame klassen. Voor het eerst werd klasse 23 überhaupt voorspeld. De score was laag, maar het werkte! Het model zag de klasse, maar kwam nog niet boven de standaard 0.5-grens. 

In Experiment 15 verbeterde ik dit doordat ik het model normaal trainde en achteraf per klasse de beslissingsdrempel optimaliseerde. Dit bleek zeer effectief. Macro-F1 steeg opnieuw, en meerdere probleemklassen werden ineens stabiel herkend. 

Laatste stap: stabiliteit en consensus 

In Experiment 16 combineerde ik threshold-optimalisatie met een ensemble van meerdere modellen. Dit dempte ruis en maakte de thresholds robuuster. Het eindpunt was Experiment 17, vijf modellen, met gebalanceerde oversampling, compacte architectuur, learning-rate decay en per-klasse geoptimaliseerde thresholds. 

Het resultaat was het beste tot nu toe: hoogste Macro-F1 (0.76), stabiele Micro-F1 (0.91), alle 31 klassen gedetecteerd. Zelfs klasse 23, die in elk vroeg experiment F1=0 had, werd nu herkend (F1=0.13) 

Eindconclusie 

Dit onderzoek begon als een zoektocht naar “het beste model”, maar eindigde in een beslissingsdynamiek. Het probleem was nooit alleen data, of embeddings, of architectuur. Het echte probleem was dat zeldzame klassen structureel werden overschaduwd, niet omdat het model ze niet kon leren, maar omdat dominante klassen altijd de voorspelling opeisten. De doorbraak kwam niet door nóg een netwerk, maar door: 

balans in representatie 

begrensde oversampling 

en vooral: per-klasse beslissingsgrenzen 

 

 