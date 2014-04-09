#include<process_data.hpp>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<iostream>
#include<fstream>

/*data dimensions are 3004x480189*/
const int rows = 3004;
const int cols = 480189;

typedef Eigen::Triplet<int> T;
std::vector<T> tripletList;

void process_netflix(const std::string& filename)
{
    std::ifstream infile(filename.c_str()); 
    std::ofstream outfile;
    outfile.open("f");
   // MatrixXd X = MatrixXd::Zero(2, 2);
        
    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    float i, j, v_ij;
    iss >> i >> j >> v_ij;

    for (int m=0; m<3048; ++m)
    {
        for (int n=0; n<480189; ++n)
        {
            if ((m+1) == i && (n+1) == j) 
            {
                outfile << v_ij << "\t";
                std::getline(infile, line);
                iss.str(line);
                iss >> i >> j >> v_ij;
            }
            else
                outfile << "0.0000\t";
        }
        outfile << "\n";
    }
    outfile.close();
}

