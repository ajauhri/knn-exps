#ifndef COVER_TREE_INCLUDED
#define COVER_TREE_INCLUDED
#include<boost/shared_ptr.hpp>
#include<vector>
#include<map>
#include<Eigen/Dense>

struct tree_node
{
    Eigen::VectorXd point;
    int max_scale;
    int min_scale;
    boost::shared_ptr<tree_node> parent;

    // scale/level wise list of children
    std::map<int, std::vector<boost::shared_ptr<tree_node>>> children;
    tree_node(Eigen::VectorXd p) : point(p), max_scale(0), min_scale(0) {}
};
typedef boost::shared_ptr<tree_node> p_tree_node;

struct ds_node
{
    p_tree_node node;
    float dist;

    ds_node(p_tree_node& p, float d) : node(p), dist(d) {}
};
typedef boost::shared_ptr<ds_node> p_ds_node;

#endif

