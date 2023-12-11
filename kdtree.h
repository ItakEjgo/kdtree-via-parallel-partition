#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <stack>
#include <queue>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <utility>
#include <iomanip>
#include "parallel.h"
#include "get_time.h" 

using namespace std;
using namespace parlay;

namespace KD_Tree{
	constexpr double eps = 1e-9;
	constexpr double INF = numeric_limits<double>::max();
	size_t n, d, k, q;
	vector<vector<double> > p_data, p_data2;
	inline int dcmp(const double &x){
 	   return fabs(x) < eps ? 0 : (x > 0 ? 1 : -1);
	}

	struct Tree_Node{
		vector<double> v;
		vector<pair<double, double> > b_box;
		// double v[d];
		// pair<double, double> b_box[d];
		size_t disc;
		Tree_Node* p;
		Tree_Node* lson;
		Tree_Node* rson;
		Tree_Node(vector<double> a, size_t b){
			v = a;
			disc = b;
			p = nullptr;
			lson = nullptr;
			rson = nullptr;
		}
	};	

	class Tree{
	private:
    	vector<vector<double> > p_set, after_partition;
    	vector<double> query_p;
    	vector<pair<double, double> > b_box;
    	Tree_Node* root;
    	size_t d = 0;
    	priority_queue<double> pq;
    	size_t discriminator = 0;
		// par partition
		size_t sort_block_size = 1000;
		size_t scan_block_size = 10000;
		size_t part_block_size = 10000;
		uint64_t rand_seed = 666888233;
		size_t *flag, *flag1, *flag2;

		// random hash
		inline uint64_t hash64(uint64_t u) {
			uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
			v = (v + 76458734123ul) + (v << 8);
			v = (v^12398981237ul) + (v >> 12);
			v = (v + 182736871ul) ^ (v >> 24);
			v = (v + 998823476ul) + (v << 17);
			v = (v ^ 27846510574ul) ^ (v >> 6);
			return v;
		}
		
		// parallel scan
		template<class Func>
		inline void scan_up(Func f, size_t s, size_t t){
			if (t - s <= scan_block_size){
				size_t ret = 0, ret2 = 0;
				for (size_t i = s; i < t; i++){
					auto tmp = f(i);
					if (tmp > 0) ret++;
					else ret2 -= tmp;
				}
				flag[t - 1] = ret;
				flag2[t - 1] = ret2;
				return;
			}
			auto f1 = [&]() {scan_up(f, s, (s + t) / 2); };
			auto f2 = [&]() {scan_up(f, (s + t) / 2, t); };
			par_do(f1, f2);
			flag[t - 1] += flag[(s + t) / 2 - 1];
			flag2[t - 1] += flag2[(s + t) / 2 - 1];
			return;
		}

		template<class Func>
		inline void scan_down(Func f, size_t s, size_t t, size_t offset, size_t offset2){
			if (t - s <= scan_block_size){
				size_t total = offset, total2 = offset2;
				for (size_t i = s; i < t; i++){
					auto tmp = f(i);
					flag[i] = total;
					flag2[i] = total2;
					if (tmp > 0) total += tmp;
					else total2 -= tmp;
				}
				return;
			}
			auto lsum = flag[(s + t) / 2 - 1];
			auto lsum2 = flag2[(s + t) / 2 - 1];
			auto f1 = [&]() {scan_down(f, s, (s + t) / 2, offset, offset2); };
			auto f2 = [&]() {scan_down(f, (s + t) / 2, t, offset + lsum, offset2 + lsum2); };
			par_do(f1, f2);
		}
		
		template<class Func>
		inline pair<size_t, size_t> scan(Func f, size_t s, size_t t){
			scan_up(f, s, t);
			size_t ret = flag[t - 1];
			size_t ret2 = flag2[t - 1];
			scan_down(f, s, t, static_cast<size_t>(0), static_cast<size_t>(0));
			return make_pair(ret, ret2);
		}

		// sequential partition
		template<class T>
		size_t seq_partition(vector<T> &A, size_t l, size_t r, T pivot_val, size_t disc_d){
 			size_t ret_l = l, ret_r = r - 1;
 			for (auto i = l; i < r; i++){
 				if (dcmp(A[i][disc_d] - pivot_val[disc_d]) < 0){
 					after_partition[ret_l++] = A[i];
 				}
 				else if (dcmp(A[i][disc_d] - pivot_val[disc_d]) > 0){
 					after_partition[ret_r--] = A[i];
 				}
 			}
 			parallel_for(l, r, [&](size_t i){
 				if (i < ret_l || i > ret_r){
 					A[i] = after_partition[i];
 				}
 				else{
 					A[i] = pivot_val;
 				}
 			});
 			// return make_pair(ret_l, ret_r + 1);
 			return (ret_l + ret_r) >> 1;
 		}

		// parallel partition
		inline size_t par_partition(vector<vector<double> > &A, size_t l, size_t r, vector<double> pivot_val, size_t disc_d){
			if (r - l <= part_block_size){
				return seq_partition(A, l, r, pivot_val, disc_d);
			}

			auto f1 = [&](int i){ return dcmp(A[i][disc_d] - pivot_val[disc_d]) < 0 ? 1 : (dcmp(A[i][disc_d] - pivot_val[disc_d]) > 0 ? -1 : 0); };
			auto ret_num = scan(f1, l, r);
			flag[r] = ret_num.first;
			flag2[r] = ret_num.second;

			parallel_for (l, r, [&](size_t i){
				if (A[i][disc_d] < pivot_val[disc_d]){
					after_partition[l + flag[i + 1] - 1] = A[i];
				}
				else if (A[i][disc_d] > pivot_val[disc_d]){
					after_partition[r - ret_num.second - 1 + flag2[i + 1]] = A[i];
				}
			});

			parallel_for(l, r, [&](size_t i){
				if (i < l + ret_num.first || i >= r - ret_num.second){
					A[i] = after_partition[i];
				}
				else{
					A[i] = pivot_val;
				}
			});
			return (l + ret_num.first + r - ret_num.second) >> 1;
		}

		inline size_t rand_index(size_t l, size_t r){
			auto id = hash64(rand_seed) % (r - l) + l;
			rand_seed = id;
			return id;
		}

    	bool is_bbox_contains_point(vector<pair<double, double> > &cur_bbox){
        	for (size_t i = 0; i < d; i++){
            	if (dcmp(query_p[i] - cur_bbox[i].first) < 0 || dcmp(query_p[i] - cur_bbox[i].second) > 0) return false;
        	}
        	return true;
    	}

    	double bbox_point_sqr_distance(vector<pair<double, double> > &cur_bbox){
        	double ret = 0.0;
        	if (!is_bbox_contains_point(cur_bbox)){
            	for (size_t i = 0; i < d; i++){
                	double dl = cur_bbox[i].first - query_p[i];
                	if (dcmp(dl) <= 0) dl = 0;
                	double dr = query_p[i] - cur_bbox[i].second;
                	if (dcmp(dr) <= 0) dr = 0;
                	double di = dcmp(dl - dr) >= 0 ? dl : dr;
                	ret += di * di;
            	}
        	}
        	return ret;
    	}

    	double point_point_sqr_distance(vector<double> &cur_p){
        	double ret = 0.0;
        	for (size_t i = 0; i < d; i++){
            	ret += (cur_p[i] - query_p[i]) * (cur_p[i] - query_p[i]);
        	}
        	return ret;
    	}

    	Tree_Node* construct_node(const size_t &l, const size_t &r, const size_t &disc_d){
        	if (l >= r) return nullptr;

        	discriminator = disc_d;
			auto mid = (l + r) >> 1;
			// mid = seq_partition(p_set, l, r, p_set[mid], disc_d);
        	nth_element(p_set.begin() + l, p_set.begin() + mid, p_set.begin() + r, [&] (vector<double> &x, vector<double> &y){
            	return dcmp(x[discriminator] - y[discriminator]) < 0;
        	});

        	Tree_Node* cur_node = new Tree_Node(p_set[mid], disc_d);
        	cur_node->b_box = b_box;

        	auto back_r = b_box[disc_d].second;
        	b_box[disc_d].second = p_set[mid][disc_d];
        	cur_node->lson = construct_node(l, mid, (disc_d + 1) % d);
        	b_box[disc_d].second = back_r;
        	if (cur_node->lson != nullptr){
            	cur_node->lson->p = cur_node;
        	}

        	auto back_l = b_box[disc_d].first;
        	b_box[disc_d].first = p_set[mid][disc_d];
        	cur_node->rson = construct_node(mid + 1, r, (disc_d + 1) % d);
        	b_box[disc_d].first = back_l;
        	if (cur_node->rson != nullptr){
            	cur_node->rson->p = cur_node;
        	}
        	return cur_node;
    	}

    	Tree_Node* construct_node_with_bbox(const size_t &l, const size_t &r, const size_t &disc_d, vector<pair<double, double> > cur_bbox){
        	if (l >= r) return nullptr;

        	discriminator = disc_d;
			auto mid = (l + r) >> 1;
			// auto mid = rand_index(l, r);
			// mid = seq_partition(p_set, l, r, p_set[mid], disc_d);
			mid = par_partition(p_set, l, r, p_set[mid], disc_d);
        	// nth_element(p_set.begin() + l, p_set.begin() + mid, p_set.begin() + r, [&] (vector<double> &x, vector<double> &y){
            // 	return dcmp(x[discriminator] - y[discriminator]) < 0;
        	// });
			
			// for (size_t i = l; i < r; i++){
			// 	after_partition[i] = p_set[i];
			// }
        	// nth_element(after_partition.begin() + l, after_partition.begin() + mid, after_partition.begin() + r, [&] (vector<double> &x, vector<double> &y){
            // 	return dcmp(x[discriminator] - y[discriminator]) < 0;
        	// });
			// seq_partition(p_set, l, r, after_partition[mid], disc_d);

        	Tree_Node* cur_node = new Tree_Node(p_set[mid], disc_d);
        	cur_node->b_box = cur_bbox;
			auto lson_bbox(cur_bbox);
			auto rson_bbox(cur_bbox);
        	lson_bbox[disc_d].second = p_set[mid][disc_d];
			rson_bbox[disc_d].first = p_set[mid][disc_d];
        	// cur_node->lson = construct_node_with_bbox(l, mid, (disc_d + 1) % d, lson_bbox);
        	// cur_node->rson = construct_node_with_bbox(mid + 1, r, (disc_d + 1) % d, rson_bbox);
        	auto f1 = [&] {cur_node->lson = construct_node_with_bbox(l, mid, (disc_d + 1) % d, lson_bbox);};
        	auto f2 = [&] {cur_node->rson = construct_node_with_bbox(mid + 1, r, (disc_d + 1) % d, rson_bbox);};
			par_do(f1, f2);
        	if (cur_node->rson != nullptr){
            	cur_node->rson->p = cur_node;
        	}
        	if (cur_node->lson != nullptr){
            	cur_node->lson->p = cur_node;
        	}
        	return cur_node;
    	}

	public:
    	void print_node(Tree_Node* cur_node){
        	for (size_t i = 0; i < d; i++){
            	cout << cur_node->v[i] << " ";
        	}
        	cout << endl;
        	if (cur_node->lson != nullptr){
            	print_node(cur_node->lson);
        	}
        	if (cur_node->rson != nullptr){
            	print_node(cur_node->rson);
        	}
    	}
    	void print_tree(){
        	print_node(root);
    	}

    	void seq_construct(vector<vector<double> > &data){
        	p_set = data;
			after_partition = data;
        	d = data[0].size();
        	b_box.resize(d);
        	for (size_t i = 0; i < d; i++){
            	b_box[i].first = data[0][i];
        	}
        	for (auto &p: data){
            	for (size_t i = 0; i < d; i++){
                	b_box[i].first = dcmp(p[i] - b_box[i].first) < 0 ? p[i] : b_box[i].first;
                	b_box[i].second = dcmp(p[i] - b_box[i].second) > 0 ? p[i] : b_box[i].second;
            	}
        	}
			parlay::timer t;
        	root = construct_node(0, p_set.size(), 0);
			t.stop();
      		std::cout << "sequential k-d tree construction time: " << t.total_time() << std::endl;
        	// root = construct_node_with_bbox(0, p_set.size(), 0, b_box);
    	}

    	void par_construct(vector<vector<double> > &data){
        	p_set = data;
			after_partition = data;
        	d = data[0].size();
        	b_box.resize(d);
			flag = (size_t*)malloc((n + 1) * sizeof(size_t));
			flag2 = (size_t*)malloc((n + 1) * sizeof(size_t));
        	for (size_t i = 0; i < d; i++){
            	b_box[i].first = data[0][i];
        	}
        	for (auto &p: data){
            	for (size_t i = 0; i < d; i++){
                	b_box[i].first = dcmp(p[i] - b_box[i].first) < 0 ? p[i] : b_box[i].first;
                	b_box[i].second = dcmp(p[i] - b_box[i].second) > 0 ? p[i] : b_box[i].second;
            	}
        	}
        	// root = construct_node(0, p_set.size(), 0);
			parlay::timer t;
        	root = construct_node_with_bbox(0, p_set.size(), 0, b_box);
			t.stop();
      		std::cout << "parallel k-d tree construction time: " << t.total_time() << std::endl;
			free(flag);
			free(flag2);
    	}

    	Tree(){}
    	~Tree(){}

    	void update_KNN_results(vector<double> &cur_p, size_t &k, double &dis_limit){
        	double cur_dis = sqrt(point_point_sqr_distance(cur_p));
        	if (dcmp(cur_dis - dis_limit) < 0){
            	pq.push(cur_dis);
            	if (pq.size() > k) pq.pop();
            	if (pq.size() == k){
                	dis_limit = dcmp(pq.top() - dis_limit) < 0 ? pq.top() : dis_limit;
            	}
        	}
    	}

    	void KNN_search(Tree_Node* cur_node, size_t &k, double &dis_limit){
        	double dis_l = INF, dis_r = INF;
        	double cur_min_bbox_dis = sqrt(bbox_point_sqr_distance(cur_node->b_box));
        	if (dcmp(cur_min_bbox_dis - dis_limit) > 0) return;

        	if (cur_node->lson != nullptr){
            	dis_l = bbox_point_sqr_distance(cur_node->lson->b_box);
        	}
        	if (cur_node->rson != nullptr){
            	dis_r = bbox_point_sqr_distance(cur_node->rson->b_box);
        	}
        	if (dcmp(dis_l - INF) != 0 || dcmp(dis_r - INF) != 0){
            	if (dcmp(dis_l - dis_r) <= 0){
                	KNN_search(cur_node->lson, k, dis_limit);
                	if (cur_node->rson != nullptr){
                    	KNN_search(cur_node->rson, k, dis_limit);
                	}
            	}
            	else {
                	KNN_search(cur_node->rson, k, dis_limit);
                	if (cur_node->lson != nullptr){
                    	KNN_search(cur_node->lson, k, dis_limit);
                	}
            	}
        	}
        	update_KNN_results(cur_node->v, k, dis_limit);
    	}

    	double KNN(vector<double> &p, size_t &k){
        	query_p = p;
        	double dis_limit = INF;
        	while (!pq.empty()) pq.pop();
        	KNN_search(root, k, dis_limit);
        	return pq.top();
    	}
	};
	
	void solve(){
    	ios::sync_with_stdio(0); cin.tie(0);
    	cin >> n >> d;
    	vector<double> cur_p(d);
    	for (size_t i = 0; i < n; i++){
        	for (size_t j = 0; j < d; j++){
            	cin >> cur_p[j];
        	}
        	p_data.emplace_back(cur_p);
			p_data2.emplace_back(cur_p);
    	}

    	Tree T, T2;

    	T2.par_construct(p_data2);
    	T.seq_construct(p_data);
    	cin >> q;
    	vector<double> query(d);
    	for (size_t i = 0; i < q; i++){
        	cin >> k;
        	k = min(k, n);
        	for (size_t j = 0; j < d; j++){
            	cin >> query[j];
        	}
        	auto res = T.KNN(query, k);
			auto res2 = T2.KNN(query, k);
        	cout << fixed << setprecision(8) << res << endl;
        	cout << fixed << setprecision(8) << res2 << endl;
    	}
	}
}