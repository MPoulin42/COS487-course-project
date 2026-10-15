import time, pipeline, json
pipeline.build_system("data/JSON Files","index_cache.pkl")
qs=pipeline.load_queries("data/Study.json")[:25]
for model in ("bm25","tfidf"):
    old={}
    for l in open(f"Retrieval Results/NJMB_paragraph_{model}_study.tsv"):
        q,_,d,r,sc,_=l.split(); old.setdefault(q,[]).append(d)
    t=time.time(); bad=0
    for q in qs:
        new=[h["id"] for h in pipeline.api.search(q["query"],model=model,level="paragraph",k=100)]
        bad+= new!=old[q["query_id"]]
    print(model,"mismatched queries:",bad,"of",len(qs),"avg s",round((time.time()-t)/len(qs),2))
