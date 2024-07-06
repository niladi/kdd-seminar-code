rsync -aPzh ~/workspace/seminar-kdd/code icarus-copy:~/clit_recommender
ssh icarus-copy "sed -i 's/\/Users\/niladi\/workspace\/seminar-kdd/\/local\/users\/uduui\/clit_recommender/' 'clit_recommender/code/src/clit_recommender/config.py'" 
#rsync -aPzh ~/workspace/seminar-kdd/data/cache icarus-copy:~/clit_recommender/data/cache
