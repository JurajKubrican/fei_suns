# fei_suns

## nastavenia    *source_dir - zdojovy priecinok obsahujuci dataset
    *cache_dir - priecinok kde sa bude ukladat cache
    *output_dir - vysleok sa ulozi sem

    
    *train - percenta trenovacich
    *test - percenta testovacich
    *valid - percenta validacnych
    
## funkcie
### compact_folders()
- nacita obazky grayscale
- znormalizuje ich
- randomizuje ich poradie
- ulozi do /cachce/a.pickle
  
### letter_pickle_preview
- nacita pickle jendotlivych labelov
- zobrazi 3 nahodnych jedincov z kazdeho labelu

### pickle_letters_to_dataset
- nacita pickle jednotlivych labelov
- da vsetky do jedneho pola
- randomizuje ich
- rozdeli podla test, train a valid (percentualne)
- ulozi na disk


### read_data
- precita vysledny piclke
- zobrazi prvych 10 z test, train a valid


