* Воспроизведение анализа в SPSS (синтаксис). Файл trade_shares.sav лежит в папке data/. Измените путь в GET FILE.
* ВНИМАНИЕ: синтаксис составлен по стандартным командам SPSS, но в среде SPSS не запускался; цифры сверяйте с results/05_agglomeration_schedule.csv.
GET FILE='data/trade_shares.sav'.
* 1) Z-оценки (создаст переменные Z + имя).
DESCRIPTIVES VARIABLES=Metals Machinery AutoParts Electronics Chemicals Dairy Meat SugarOilFeed WoodFurniture Pharma
  /SAVE.
* 2) Иерархический кластерный анализ: метод Уорда, квадрат евклидова расстояния.
CLUSTER ZMetals ZMachinery ZAutoParts ZElectronics ZChemicals ZDairy ZMeat ZSugarOilFeed ZWoodFurniture ZPharma
  /METHOD WARD
  /MEASURE=SEUCLID
  /ID=Country
  /PRINT SCHEDULE CLUSTER(2,8)
  /PLOT DENDROGRAM VICICLE
  /SAVE CLUSTER(5).
* 3) Профиль кластеров: средние по переменным и ANOVA.
ONEWAY Metals Machinery AutoParts Electronics Chemicals Dairy Meat SugarOilFeed WoodFurniture Pharma BY CLU5_1
  /STATISTICS DESCRIPTIVES
  /MISSING ANALYSIS.
* 4) Проверка K-Means на 5 кластерах (начальные центры SPSS выбирает сам; в Python-версии стартовали от центров Уорда).
QUICK CLUSTER ZMetals ZMachinery ZAutoParts ZElectronics ZChemicals ZDairy ZMeat ZSugarOilFeed ZWoodFurniture ZPharma
  /CRITERIA=CLUSTER(5) MXITER(20) CONVERGE(0)
  /METHOD=KMEANS(NOUPDATE)
  /PRINT INITIAL ANOVA CLUSTER DISTAN.
