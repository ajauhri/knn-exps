#include<iostream>
#include<vector>
#include<process_data.hpp>
#include<cover_tree.hpp>
#include<algorithm>
#include<random>

using Eigen::MatrixXd;
const float g_scale = 1.3;
float il2 = 1./ log(g_scale);
std::mt19937 gen;

// distance are to be presented as 2^i where i is the representative for scale
inline int get_scale(float d)
{
    return (int) ceilf(il2 * log(d));
}

unsigned int rand(unsigned int a, unsigned int b)
{
    std::uniform_int_distribution<> dis(a,b);
    return dis(gen);
}

// considers two input vectors in some dimension i.e. of equal size
inline float distance(const Eigen::VectorXd& p1, const Eigen::VectorXd& p2)
{
    assert(p1.size() == p2.size());
    float sum = 0.0f;
    for (int i=0; i<p1.size(); i++)
    {
        float d1 = p1(i) - p2(i);
        d1 *= d1;
        sum += d1;
    }
    return sqrt(sum);
}

// creates a tree node
p_tree_node create_tree_node(const Eigen::VectorXd& p)
{
    p_tree_node leaf(new tree_node(p));
    return leaf;
}

// gets all children at a scale and returns a vector with all points at that scale and distance from a given point (p). Distances stored are not scaled using `get_scale`
void get_children(const Eigen::VectorXd& p, std::vector<p_ds_node>& Qi_p_ds, int scale)
{
    for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
    {
        if (Qi_p_ds[i]->node->children.find(scale) != Qi_p_ds[i]->node->children.end()) //if not empty key slot
        {
            for (unsigned int j=0; j<Qi_p_ds[i]->node->children[scale].size(); ++j)
            {
                p_tree_node& node = Qi_p_ds[i]->node->children[scale][j];
                p_ds_node ds(new ds_node(node, distance(node->point, p)));
                Qi_p_ds.push_back(ds);
            }
        }
    }
}

// return the min distance
float get_min(std::vector<p_ds_node>& Qi_p_ds)
{
    float min = pow(g_scale, 100);
    for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
    {
        if (Qi_p_ds[i]->dist < min)
            min = Qi_p_ds[i]->dist;
    }
    return min;
}

// return the min node
p_tree_node get_min_node(std::vector<p_ds_node>& Qi_p_ds)
{
    float min = pow(g_scale, 100);
    p_tree_node min_node;
    for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
    {
        if (Qi_p_ds[i]->dist < min)
        {
            min = Qi_p_ds[i]->dist;
            min_node = Qi_p_ds[i]->node;
        }
    }
    return min_node;
}


bool compare(const p_ds_node& a, const p_ds_node& b)
{
    return (a->dist < b->dist);
}

void knn(int k, const Eigen::VectorXd& p, p_tree_node& root)
{
    std::vector<p_ds_node> Qi_p_ds;
    p_ds_node ds(new ds_node(root, distance(root->point, p)));
    Qi_p_ds.push_back(ds);
    for(int i = root->max_scale; i >= root->min_scale; --i)
    {
        get_children(p, Qi_p_ds, i);
        sort(Qi_p_ds.begin(), Qi_p_ds.end(), compare);
        float d_p_Q = Qi_p_ds[k-1]->dist;
    
        float scale_dist = pow(g_scale, i);
        for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
        {
            if (Qi_p_ds[i]->dist > scale_dist + d_p_Q)
                Qi_p_ds.erase(Qi_p_ds.begin() + i);
        }
    }
    sort(Qi_p_ds.begin(), Qi_p_ds.end(), compare);
}

p_tree_node insert(const Eigen::VectorXd& p, p_tree_node& root, int max_scale)
{
    int i = max_scale;
    std::vector<p_ds_node> Qi_p_ds;
    p_ds_node ds(new ds_node(root, distance(root->point, p)));
    Qi_p_ds.push_back(ds);
    int pi;
    p_tree_node parent; 
    while (1)
    {
        // find all children at a level
        get_children(p, Qi_p_ds, i);
        float min_d_p_Q = get_min(Qi_p_ds);
    
        // point already exists, so not added again
        if (min_d_p_Q == 0.0f)
        {
            return get_min_node(Qi_p_ds);
        }

        // parent has been found 
        else if (min_d_p_Q > pow(g_scale, i))
            break;
        else 
        {
            float scale_dist = pow(g_scale, i);
            if (min_d_p_Q <= pow(g_scale, i))
            {
                while (1)
                {
                    // randomly select a parent which satisfies the invariant i.e. <= 2^i distance from point p
                    int pos = rand(0, Qi_p_ds.size() - 1);
                    if (Qi_p_ds[pos]->dist <= scale_dist)
                    {
                        parent = Qi_p_ds[rand(0, Qi_p_ds.size() - 1)]->node;
                        pi = i;
                        break;
                    }
                }
            }
            
            // construct Q_{i-1} by removing all elements for which have distance greater than 2^i i.e. to maintain cover tree invariant
            for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
            {
                if (Qi_p_ds[i]->dist > scale_dist)
                    Qi_p_ds.erase(Qi_p_ds.begin() + i);
            }
            i--;
        }
    }

    // need to check if p already in self.children?
    p_tree_node new_node = create_tree_node(p);
    parent->children[pi].push_back(new_node);
    new_node->parent = parent;
    root->min_scale = std::min(root->min_scale, pi-1);
    return new_node;
}

void remove(const Eigen::VectorXd& p, p_tree_node& root, int max_scale)
{
    int i = max_scale;
    std::vector<p_ds_node> Qi_p_ds;
    p_ds_node ds(new ds_node(root, distance(root->point, p)));
    Qi_p_ds.push_back(ds);
    
    while (1)
    {
        get_children(p, Qi_p_ds, i);
        float min_d_p_Q = get_min(Qi_p_ds);

        // found
        if (min_d_p_Q == 0.0f)
        {
            p_tree_node tbr_node = get_min_node(Qi_p_ds);

            // unlink the node from its parent
            if (tbr_node->parent->children[i].size() == 1)
                tbr_node->parent->children.erase(i);
            else 
            {
                for(unsigned int j=0; j<tbr_node->parent->children[i].size(); ++j)
                {
                    if (tbr_node->parent->children[i][j]->point == p)
                    {
                        tbr_node->parent->children[i].erase(tbr_node->parent->children[i].begin() + j);
                        break;
                    }
                }
            }
            
            // insert all children of to be deleted node, p, into the cover tree
            typedef std::map<int, std::vector<boost::shared_ptr<tree_node>>>::iterator iter;
            for (iter it = tbr_node->children.begin(); it != tbr_node->children.end(); ++it)
            {
                for (unsigned int j=0; j < it->second.size(); ++j)
                {
                    p_tree_node n = insert(it->second[j]->point, root, root->max_scale);
                    // and don't forget to link the grand-children 
                    n->children = it->second[j]->children;
                }
                tbr_node->children.erase(it);
            }
            assert(tbr_node->children.size() == 0);
            tbr_node->parent = nullptr;

        }
        else if (min_d_p_Q > pow(g_scale, i))
        {
            // is it there?
            break;
        }
        else
        {
            float scale_dist = pow(g_scale, i);
            for (unsigned int i=0; i<Qi_p_ds.size(); ++i)
            {
                if (Qi_p_ds[i]->dist > scale_dist)
                    Qi_p_ds.erase(Qi_p_ds.begin() + i);
            }
            i--;
        }
    }
} 

int main(int argc, char **argv)
{
    // currently this method is parsing based and at all dynamic to consider many kinds of files.
    // it would be preferred to have a generic parser... 
    process_netflix(std::string("../data/netflix.dat")); 
    std::cout<<"parsing done\n";
    //std::cout<<"total points="<<X.rows()<<"\n";
    
    float max_dist = 0.0f;
    /*
    for (int i=1; i<X.rows(); ++i)
    {
        float dist = distance(X.row(0), X.row(i));
        max_dist = max_dist < dist ? dist : max_dist;
        std::cout<<X.row(i)[0]<<" "<<X.row(i)[1]<<" "<<X.row(i)[2]<<"\n";
    }
    std::cout<<"************\n";
    p_tree_node root = create_tree_node(X.row(0));
    root->max_scale = get_scale(max_dist);
    root->min_scale = root->max_scale;

    for (int i = 1; i<X.rows(); ++i)
        insert(X.row(i), root, root->max_scale);

    p_tree_node temp = root;    

    // here again this has data specific columns and it makes the usage limited. Need to expand here too...
    //p_tree_node top = batch_create(X);
    std::cout<<"all points inserted\n";
    std::vector<p_tree_node> queue;
    queue.push_back(root);
    int s = 1;
    while (queue.size() > 0)
    {
        typedef std::map<int, std::vector<boost::shared_ptr<tree_node>>>::iterator iter;
        for (iter i = queue[0]->children.begin(); i != queue[0]->children.end(); ++i)
        {
            for(unsigned int j=0; j < i->second.size(); ++j)
                queue.push_back(i->second[j]);
            s += i->second.size();
            
        }
        queue.erase(queue.begin());
    }
    */
    return 0;
}

