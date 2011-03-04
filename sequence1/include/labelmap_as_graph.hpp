#include <utility>

template <class T>
struct labelmap_graph_adjacency_iterator;

template <class T>
struct labelmap_graph_io_edge_iterator;

template <class T>
struct labelmap_graph_vertex_iterator;

template <class T>
struct labelmap_graph_edge_iterator;


template <class T>
struct labelmap_graph_t
{
    typedef int vertex_descriptor;
    //typedef labelmap_graph_vertex_iterator<T> vertex_iterator;
    typedef int* vertex_iterator;
//typedef int const* 
    typedef std::pair<vertex_descriptor, vertex_descriptor> edge_descriptor;
    typedef labelmap_graph_edge_iterator<T> edge_iterator;
    typedef labelmap_graph_adjacency_iterator<T> adjacency_iterator;
    typedef labelmap_graph_io_edge_iterator<T> out_edge_iterator;
    typedef labelmap_graph_io_edge_iterator<T> in_edge_iterator;

    typedef undirected_tag directed_category;
    typedef disallow_parallel_edge_tag edge_parallel_category;

    typedef boost::no_property vertex_property_type;
#if 0
    struct lbm_traversal_tag :
	public virtual vertex_list_graph_tag,
	public virtual incidence_graph_tag,
	public virtual adjacency_graph_tag { };
#endif

    //typedef adjacency_graph_tag traversal_category;
    //typedef vertex_list_graph_tag traversal_category;
    struct traversal_category :
	public virtual boost::vertex_list_graph_tag {};
    typedef boost::vertex_list_graph_tag graph_tag;

    typedef int vertices_size_type;
    typedef int edges_size_type;
    typedef int degree_size_type;


    labelmap_graph_t(matrix<T> & mat): mat_(&mat){}
    matrix<T> * mat_;
    int num_vertices() const { return mat_->size1()*mat_->size2(); }

    int xy_to_index(int x, int y) const {
	return x+y*mat_->size2();
    }
    void index_to_xy(int ind, int& x, int& y) const {
	y = ind/mat_->size2();
	x = ind%mat_->size2();
    }

    bool edge_exist(vertex_descriptor const & v, int dx, int dy) const {
	matrix<T> const& m=*mat_;
	int x, y;
	index_to_xy(v, x, y);
	int y2 = y+dy;
	if(y2<0 || y2>=m.size1()) return false;
	int x2 = x+dx;
	if(x2<0 || x2>=m.size2()) return false;
	return m(y, x)==m(y2, x2);
    }
    int out_degree(vertex_descriptor const& v) const;
    

    const vertex_iterator vertices_begin() const {
	return mat_->data().begin();
    }
    const vertex_iterator vertices_end() const {
	return mat_->data().end();
    }
};

namespace boost {
template <class T>
struct graph_traits<labelmap_graph_t<T> >
{
    typedef typename labelmap_graph_t<T>::traversal_category traversal_category;
    typedef typename labelmap_graph_t<T>::vertex_descriptor vertex_descriptor;
    typedef typename labelmap_graph_t<T>::vertex_iterator vertex_iterator;

    typedef typename labelmap_graph_t<T>::directed_category directed_category;
    typedef typename labelmap_graph_t<T>::edge_parallel_category edge_parallel_category;

    typedef typename labelmap_graph_t<T>::edge_descriptor edge_descriptor;

    typedef typename labelmap_graph_t<T>::out_edge_iterator out_edge_iterator;

    typedef int degree_size_type;

    //typedef typename graph_traits<labelmap_graph_t<T> >::traversal_category>::value::traversal_category;
};

#if 0
template <class T>
struct graph_traits<labelmap_graph_t<T>& >
{
    typedef typename labelmap_graph_t<T>::traversal_category traversal_category;
    //typedef typename graph_traits<labelmap_graph_t<T> >::traversal_category>::value::traversal_category;
};
#endif 

}

template <class T>
struct labelmap_graph_adjacency_iterator
{
    labelmap_graph_adjacency_iterator(int ii, int pp, const labelmap_graph_t<T> & g)
	: g_(&g), ii_(ii), pp_(pp) {
    }
    static int const Nbr[4][2];
    typename labelmap_graph_t<T>::vertex_descriptor operator*() const {
	int x, y;
	g_->index_to_xy(pp_, x, y);
	return std::make_pair(x+Nbr[ii_][0], y+Nbr[ii_][1]);
    }
    void operator++() {
	if(ii_>=4) return;
	for(++ii_; ii_<4; ++ii_)
	{
	    if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
	       break;
	}
    }
    bool operator==(labelmap_graph_adjacency_iterator const & x) const
	{ return ii_ == x.ii_; }
    void reset_begin() {
	ii_=-1;
	for(++ii_; ii_<4; ++ii_)
	{
	    if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
	       break;
	}
    }
protected:
    int ii_;
    typename labelmap_graph_t<T>::vertex_descriptor pp_;
    labelmap_graph_t<T> const* g_;
};


template <class T>
struct labelmap_graph_io_edge_iterator
{
    labelmap_graph_io_edge_iterator(){}
    labelmap_graph_io_edge_iterator(int ii, int pp, const labelmap_graph_t<T> & g)
	: g_(&g), ii_(ii), pp_(pp) {
    }
    static int const Nbr[4][2];
    typename labelmap_graph_t<T>::edge_descriptor operator*() const {
	int x, y;
	g_->index_to_xy(pp_, x, y);
	int ind = g_->xy_to_index(x+Nbr[ii_][0], y+Nbr[ii_][1]);
	return std::make_pair(pp_, ind);
    }
    void operator++() {
	if(ii_>=4) return;
	for(++ii_; ii_<4; ++ii_)
	{
	    if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
	       break;
	}
    }
    bool operator==(labelmap_graph_io_edge_iterator const & x) const
	{ return ii_ == x.ii_; }
    bool operator!=(labelmap_graph_io_edge_iterator const & x) const
	{ return ii_ != x.ii_; }
    void reset_begin() {
	ii_=-1;
	for(++ii_; ii_<4; ++ii_)
	{
	    if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
	       break;
	}
    }
    void reset_end() {
	ii_ = 4;
    }
protected:
    int ii_;
    typename labelmap_graph_t<T>::vertex_descriptor pp_;
    labelmap_graph_t<T> const* g_;
};
#if 0
template <class T>
struct labelmap_graph_vertex_iterator
{
    typedef typename labelmap_graph_t<T>::vertex_descriptor value_type;


    labelmap_graph_vertex_iterator() {}
    labelmap_graph_vertex_iterator(value_type pp,
				   labelmap_graph_t<T> const& g)
	: g_(&g) {
	index = g.vertex_to_index(pp);
    }
    labelmap_graph_vertex_iterator(int ii,
				   labelmap_graph_t<T> const& g)
	: g_(&g), index(ii) {
    }
    value_type operator*() const {
	return g_->index_to_vertex(index);
    }
    void operator++() {
	if(index<g_->num_vertices()) index++;
    }
    bool operator==(labelmap_graph_vertex_iterator const& x) const 
	{ return index==x.index;}
    bool operator!=(labelmap_graph_vertex_iterator const& x) const 
	{ return index!=x.index;}
    void reset_begin() {
	index = 0;
    }
    void reset_end() {
	index = g_->num_vertices();
    }

    labelmap_graph_t<T> const* g_;
    int index;
};
#endif

template <class T>
struct labelmap_graph_edge_iterator
{
    labelmap_graph_edge_iterator(int ii, int pp, const labelmap_graph_t<T> & g)
	: g_(&g), ii_(ii), pp_(pp) {
    }
    static int const Nbr[2][2];
    typename labelmap_graph_t<T>::edge_descriptor operator*() const {
	int x, y;
	g_->index_to_xy(pp_, x, y);
	int ind = g_->xy_to_index(x+Nbr[ii_][0], y+Nbr[ii_][1]);
	return std::make_pair(pp_, ind);
    }
    void operator++() {
	if(ii_>=2 && pp_!=g_->vertices_end()) return;
	do{
	    for(++ii_; ii_<2; ++ii_)
	    {
		if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
		    return;
	    }
	}
	while(pp_!=g_->vertices_end());
    }
    bool operator==(labelmap_graph_edge_iterator const & x) const
	{ return ii_ == x.ii_; }
    void reset_begin() {
	ii_=-1;
	pp_ = g_->vertices_begin();
	do{
	    for(++ii_; ii_<2; ++ii_)
	    {
		if(g_->edge_exist(pp_, Nbr[ii_][0], Nbr[ii_][1]) )
		    return;
	    }
	}
	while(pp_!=g_->vertices_end());

    }
protected:
    int ii_;
    typename labelmap_graph_t<T>::vertex_iterator pp_;
    labelmap_graph_t<T> const* g_;
};



template <class T>
    const int labelmap_graph_adjacency_iterator<T>::Nbr[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
template <class T>
    const int labelmap_graph_io_edge_iterator<T>::Nbr[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

template <class T>
    const int labelmap_graph_edge_iterator<T>::Nbr[2][2] = {{1, 0}, {0, 1}};

template <class T>
int num_vertices(labelmap_graph_t<T> const& g)	{ return g.num_vertices(); }

template <class T>
std::pair<typename labelmap_graph_t<T>::adjacency_iterator,
	  typename labelmap_graph_t<T>::adjacency_iterator>
adjacent_vertices(typename labelmap_graph_t<T>::vertex_descriptor v,
		  labelmap_graph_t<T> const & g) 
{
    typedef typename labelmap_graph_t<T>::adjacency_iterator Iter;
    Iter begin(0, v, g);
    begin.reset_begin();
    return std::make_pair(begin, Iter(4, v, g));
}


template <class T>
std::pair<
    typename labelmap_graph_t<T>::vertex_iterator,
    typename labelmap_graph_t<T>::vertex_iterator>
vertices(labelmap_graph_t<T> const & g)
{
    return std::make_pair(g.vertices_begin(), g.vertices_end());
}

template <class T>
std::pair<
    typename labelmap_graph_t<T>::out_edge_iterator,
    typename labelmap_graph_t<T>::out_edge_iterator>
out_edges(typename labelmap_graph_t<T>::vertex_descriptor const & v,
	  labelmap_graph_t<T> const & g)
{
    labelmap_graph_io_edge_iterator<T> b,e;
    b.reset_begin();
    e.reset_end();
    return std::make_pair(b, e);
}
template <class T>
std::pair<int, int>
get(boost::vertex_index_t, const labelmap_graph_t<T>& g)
{
    return std::make_pair(0, g.num_vertices());
}

template <class T>
int out_degree(typename labelmap_graph_t<T>::vertex_descriptor const & v,
	       const labelmap_graph_t<T>& g)
{
    return g.out_degree(v);
}

template <class T>
int labelmap_graph_t<T>::out_degree(labelmap_graph_t<T>::vertex_descriptor const& v) const
{
    int res=0;
    const int (*Nbr)[2]  = labelmap_graph_adjacency_iterator<T>::Nbr;
    for(int ii=0; ii<4; ++ii)
    {
	if(edge_exist(v, Nbr[ii][0], Nbr[ii][1])) res ++;

    }
    return res;
}
