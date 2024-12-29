rsync -aPzh ~/workspace/kit/seminar-kdd/code/src/* icarus-copy:~/clit_recommender/code/src
rsync -aPzh ~/workspace/kit/seminar-kdd/code/requirements.txt icarus-copy:~/clit_recommender/code/requirements.txt
rsync -aPzh ~/workspace/kit/seminar-kdd/code/data/best_graphs/* icarus-copy:~/clit_recommender/code/data/best_graphs
rsync -aPzh ~/workspace/kit/seminar-kdd/code/data/embeddings/* icarus-copy:~/clit_recommender/code/data/embeddings
ssh icarus-copy "sed -i 's/\/Users\/niladi\/workspace\/kit\/seminar-kdd/\/local\/users\/uduui\/clit_recommender/' 'clit_recommender/code/src/clit_recommender/__init__.py'" 
ssh icarus-copy "sed -i 's/cpu/cuda:1/' 'clit_recommender/code/src/clit_recommender/config.py'" 
#rsync -aPzh ~/workspace/seminar-kdd/data/cache icarus-copy:~/clit_recommender/data/cache
