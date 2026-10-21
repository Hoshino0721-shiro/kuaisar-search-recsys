import unittest
from src.v2.recall_fusion import route_limits, select_recall
from src.v2.cascade import Budget, Candidate, Cascade


class RecallFusionTests(unittest.TestCase):
    def test_default_rrf_matches_independent_reference_and_zero_disables_route(self):
        ranks = {(10, 'a'): 1, (20, 'a'): 2, (20, 'b'): 3, (30, 'b'): 1}
        scores = {10: 1/61, 20: 1/62+2/63, 30: 2/61}
        self.assertEqual(select_recall(ranks, 3, {'a':1, 'b':2}),
                         sorted(scores, key=lambda i: (-scores[i],i)))
        self.assertEqual(select_recall(ranks, 3, {'a':1, 'b':0}), [10,20])

    def test_quota_counts_unique_additions_and_fills_missing_capacity(self):
        ranks = {(10,'recent'):1,(20,'recent'):2,(30,'recent'):3,(40,'recent'):4,
                 (10,'itemcf'):1,(50,'itemcf'):2,(60,'itemcf'):3}
        policy = {'method':'quota','allocations':[['recent',2],['itemcf',1]],
                  'fill_order':['recent']}
        self.assertEqual(select_recall(ranks,4,policy=policy),[10,20,50,30])
        cold = {key:r for key,r in ranks.items() if key[1]=='recent'}
        self.assertEqual(select_recall(cold,4,policy=policy),[10,20,30,40])

    def test_quota_enforces_availability_exclusions_and_source_traces_in_cascade(self):
        routes = [Candidate(10,'recent',1,0),Candidate(20,'recent',2,0),
                  Candidate(30,'recent',3,0),Candidate(40,'itemcf',1,101),
                  Candidate(50,'itemcf',2,0),Candidate(60,'itemcf',3,0)]
        policy = {'method':'quota','allocations':[['recent',2],['itemcf',1]],
                  'fill_order':['recent']}
        score = lambda u,ids,t:[0]*len(ids)
        result = Cascade(score,score,Budget(3,3,3,3)).run(
            1,100,routes,exclude_items={50},route_weights={'itemcf':0},fusion_policy=policy)
        self.assertEqual(result['stages'][0]['items'],[10,20,60])
        self.assertEqual(result['candidate_sources']['60'],['itemcf'])
        self.assertEqual(result['rejected'],{'not_yet_available':1,'excluded_item':1})

    def test_route_limits_keep_legacy_defaults_allow_zero_and_reject_invalid_configs(self):
        self.assertEqual(route_limits({'route_k':1000}),
                         {'popular':1000,'recent':1000,'itemcf':1000})
        self.assertEqual(route_limits({'route_limits':{'popular':0,'recent':2000}}),
                         {'popular':0,'recent':2000,'itemcf':1000})
        self.assertEqual(route_limits({'route_k':5,'route_limits':{'tower':3}}),
                         {'popular':5,'recent':5,'itemcf':5,'tower':3})
        for cfg in [{'route_limits':{'recent':True}}, {'route_limits':{'recent':-1}},
                    {'route_k':0}, {'route_limits':{'typo':100}}]:
            with self.assertRaises(ValueError):
                route_limits(cfg)

    def test_policy_budget_ties_and_fail_fast(self):
        self.assertEqual(select_recall({(20,'a'):1,(10,'b'):1},1),[10])
        bad = [{'method':'typo'}, {'method':'quota','allocations':[['a',2],['b',2]]},
               {'method':'quota','allocations':[['a',1],['a',1]]},
               {'method':'quota','allocations':[['a',True]]},
               {'method':'quota','fill_order':['a','a']}]
        for policy in bad:
            with self.assertRaises(ValueError):
                select_recall({},3,policy=policy)
        for weights in [{'a':float('nan')},{'a':-1}]:
            with self.assertRaises(ValueError):
                select_recall({},3,weights)


if __name__ == '__main__':
    unittest.main()
