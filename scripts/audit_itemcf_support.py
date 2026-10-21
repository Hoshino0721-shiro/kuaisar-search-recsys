"""Check sampled real graph edges against binary training co-occurrences."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.itemcf_index import DATA, IndexConfig, ItemCFIndex, read_snapshot, digest


def audit(path, output, data):
    if output.exists():
        raise FileExistsError('Refusing to overwrite graph audit')
    index=ItemCFIndex(path, digest(data/'manifest.json'))
    try:
        config=index.config
        if config.user_iuf or config.graph_half_life_days or config.shrinkage:
            raise ValueError('This independent count audit expects unweighted, unshrunk graph')
        snapshot=read_snapshot(data,config)
        binary=csr_matrix((np.ones(len(snapshot.rows)),(snapshot.rows,snapshot.columns)),
                          shape=(len(snapshot.users),len(snapshot.items)))
        csc=binary.tocsc()
        degree=np.asarray(binary.sum(axis=0)).ravel()
        nonempty=np.flatnonzero(np.any(index.neighbors>=0,axis=1))
        rng=np.random.default_rng(2026)
        sample=np.unique(np.r_[rng.choice(len(index.items),min(256,len(index.items)),replace=False),
                               rng.choice(nonempty,min(256,len(nonempty)),replace=False)])
        edges=0
        for row in sample:
            valid=index.neighbors[row]>=0
            cols=index.neighbors[row,valid]
            scores=index.similarities[row,valid]
            counts=(csc[:,int(row)].T@binary).tocsr()
            counts.sort_indices()
            positions=np.searchsorted(counts.indices,cols)
            assert np.all(positions<len(counts.indices))
            np.testing.assert_array_equal(counts.indices[positions],cols)
            support=counts.data[positions]
            assert np.all(support>=config.min_cooccurrence)
            assert int(row) not in cols
            assert len(cols)==len(set(cols.tolist()))<=config.neighbors
            expected=support/(degree[row]*degree[cols])**config.popularity_alpha
            np.testing.assert_allclose(scores,expected,rtol=2e-6,atol=1e-7)
            edges+=len(cols)
        result={'status':'passed','index_version':index.metadata['version'],
                'sampled_rows':len(sample),'checked_edges':edges,'seed':2026,
                'checks':['sampled edges have required independent binary common-user count',
                          'scores match count / (item degrees product)^alpha',
                          'no self-edge or duplicate neighbor; row budget respected'],
                'scope':'sampled edge validity, not exhaustive graph or independent temporal quality',
                'test_labels_used':False,'source_sha256':digest(Path(__file__))}
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        return result
    finally:
        index.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=DATA)
    args=parser.parse_args()
    audit(args.index,args.out,args.data)
