// mesh generator for a simple quadrilateral domain

// no need to have a header for this. I will just have a function 

#include "myMeshGenerator.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept> 
#include <math.h> 
#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"
#include <boost/algorithm/string.hpp>
#include <Eigen/Dense> // most of the vector functions I will need inside of an element
#include <Eigen/Sparse> // functions for solution of linear systems
#include <Eigen/OrderingMethods>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace Eigen;

// a simple quad mesh generation, really stupid one
QuadMesh myRectangleMesh(const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution)
{
	// number of points in the x and y directions
	int n_x_points = meshResolution[0];
	int n_y_points = meshResolution[1];

	// dimensions of the mesh in the x and y direction
	double x_init = rectangleDimensions[0];
	double x_final = rectangleDimensions[1];
	double y_init = rectangleDimensions[2];
	double y_final = rectangleDimensions[3];
	int n_nodes = n_x_points*n_y_points;
	int n_elems = (n_x_points-1)*(n_y_points-1);
	std::cout<<"Going to create a mesh of "<<n_nodes<<" nodes and "<<n_elems<<" elements\n";
	std::cout<<"X0 = "<<x_init<<", XF = "<<x_final<<", Y0 = "<<y_init<<", YF = "<<y_final<<"\n";
	std::vector<Vector2d> NODES(n_nodes,Vector2d(0.,0.));
	std::vector<int> elem0 = {0,0,0,0};
	std::vector<std::vector<int> > ELEMENTS(n_elems,elem0);
	std::vector<int> BOUNDARIES(n_x_points*n_y_points,0);
	// create the nodes row by row
	for(int i=0;i<n_x_points;i++){
		for(int j=0;j<n_y_points;j++){
			double x_coord = x_init+ i*(x_final-x_init)/(n_x_points-1);
			double y_coord = y_init+ j*(y_final-y_init)/(n_y_points-1);
			NODES[j*n_x_points+i](0) = x_coord;
			NODES[j*n_x_points+i](1) = y_coord;
			if(i==0 || i==n_x_points-1 || j==0 || j==n_y_points-1){
				BOUNDARIES[j*n_x_points+i]=1;
			}
		}
	}
	std::cout<<"... filled nodes...\n";
	// create the connectivity
	for(int i=0;i<n_x_points-1;i++){
		for(int j=0;j<n_y_points-1;j++){		
			ELEMENTS[j*(n_x_points-1)+i][0] = j*n_x_points+i;
			ELEMENTS[j*(n_x_points-1)+i][1] = j*n_x_points+i+1;
			ELEMENTS[j*(n_x_points-1)+i][2] = (j+1)*n_x_points+i+1;
			ELEMENTS[j*(n_x_points-1)+i][3] = (j+1)*n_x_points+i;
		}
	}
	QuadMesh myMesh;
	myMesh.nodes = NODES;
	myMesh.elements = ELEMENTS;
	myMesh.boundary_flag = BOUNDARIES;
	myMesh.n_nodes = n_x_points*n_y_points;
	myMesh.n_elements = (n_x_points-1)*(n_y_points-1);
	return myMesh;
}


double distanceX2E(std::vector<double> &ellipse, double x_coord, double y_coord,double mesh_size)
{
	// given a point and the geometry of an ellipse, give me the
	// distance along the x axis towards the ellipse
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double x_axis = ellipse[2];
	double y_axis = ellipse[3];
	double alpha = ellipse[4];
	
	// equation of the ellipse 
	double x_ellipse_1 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 -\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	double x_ellipse_2 = (pow(x_axis,2)*x_center*pow(sin(alpha),2) + pow(x_axis,2)*y_center*sin(2*alpha)/2 - pow(x_axis,2)*y_coord*sin(2*alpha)/2 \
						+ x_center*pow(y_axis,2)*pow(cos(alpha),2) + pow(y_axis,2)*y_center*sin(2*alpha)/2 - pow(y_axis,2)*y_coord*sin(2*alpha)/2 +\
						sqrt(pow(x_axis,2)*pow(y_axis,2)*(pow(x_axis,2)*pow(sin(alpha),2) - pow(y_axis,2)*pow(sin(alpha),2) + pow(y_axis,2) \
				- 4*pow(y_center,2)*pow(sin(alpha),4) + 4*pow(y_center,2)*pow(sin(alpha),2) - pow(y_center,2) + 8*y_center*y_coord*pow(sin(alpha),4)\
				 - 8*y_center*y_coord*pow(sin(alpha),2) + 2*y_center*y_coord - 4*pow(y_coord,2)*pow(sin(alpha),4) + 4*pow(y_coord,2)*pow(sin(alpha),2)\
				  - pow(y_coord,2))))/(pow(x_axis,2)*pow(sin(alpha),2) + pow(y_axis,2)*pow(cos(alpha),2));
	// which is closer?
	double distance1 = fabs(x_ellipse_1 - x_coord);
	double distance2 = fabs(x_ellipse_2 - x_coord);
	if(distance1<distance2 && distance1<mesh_size){
		return x_ellipse_1-x_coord;
	}else if(distance2<mesh_size){
		return x_ellipse_2-x_coord;
	}
	return 0;
}

// conform the mesh to a given ellipse
void conformMesh2Ellipse(QuadMesh &myMesh, std::vector<double> &ellipse)
{
	// the ellipse is defined by center, axis, and angle
	double x_center = ellipse[0];
	double y_center = ellipse[1];
	double x_axis = ellipse[2];
	double y_axis = ellipse[3];
	double alpha_ellipse = ellipse[4];
	
	// loop over the mesh nodes 
	double x_coord,y_coord,check,d_x2e,mesh_size;
	mesh_size = (myMesh.nodes[1](0)-myMesh.nodes[0](0))/1.1;
	for(int i=0;i<myMesh.n_nodes;i++){
		// if the point is inside check if it is close, if it is in a certain 
		// range smaller than mesh size then move it the ellipse along x. ta ta
		x_coord = myMesh.nodes[i](0);
		y_coord = myMesh.nodes[i](1);
		check = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
				pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
		if(check>1){
			// calculate the distance to the ellipse along x axis
			d_x2e = distanceX2E(ellipse,x_coord,y_coord,mesh_size);
			myMesh.nodes[i](0) += d_x2e;
		}
	}					
}

// read in the Abaqus file and save the info into myMesh (BCs assignment should be manually editted, not universal)
void readAbaqusToMymesh(const char* filename, QuadMesh &myMesh){
	// READ NODES
	std::vector<Vector2d> node_X; node_X.clear();
	std::ifstream myfile(filename);
	std::string line;
	std::string keyword_node = "*Node";
	if (myfile.is_open())
	{
		// read in until you find the keyword *NODE
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_node);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs;
					boost::split(strs,line,boost::is_any_of(","));
					node_X.push_back(Vector2d(std::stod(strs[1]),std::stod(strs[2])));
      			}
      		}
    	}
    }
    myfile.close();
    myMesh.nodes = node_X;
    myMesh.n_nodes = node_X.size();
    
	// READ ELEMENTS	
	std::vector<std::vector<int> > LineQuadri; LineQuadri.clear();
	myfile.open(filename);
	std::string keyword_element = "*Element";
	if (myfile.is_open())
	{
		// read in until you find the keyword *ELEMENTS
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_element);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> elemi;elemi.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=1;nodei<strs1.size();nodei++)
					{
						//std::cout<<strs1[nodei]<<",";
						elemi.push_back(std::stoi(strs1[nodei])-1);
					}
					//std::cout<<"\n";
					LineQuadri.push_back(elemi);
      			}
      		}
    	}
    }
    myfile.close();
    myMesh.elements = LineQuadri;
    myMesh.n_elements = LineQuadri.size();

	// READ ELEMENT GROUPS
	myfile.open(filename);
	// All element indices must be expanded!!!
	std::vector<int> material(myMesh.n_elements,0);
	std::string keyword_elset = "*Elset1";
	if (myfile.is_open())
	{
		// read in until you find the keyword *Elset1
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_elset);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					int elemmat;
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						elemmat = (std::stoi(strs1[nodei])-1);
						material[elemmat] = 0;
					}
      			}
      		}
    	}
    }
    myfile.close();
    myfile.open(filename);
    keyword_elset = "*Elset2";
	if (myfile.is_open())
	{
		// read in until you find the keyword *Elset2
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_elset);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					int elemmat;
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						elemmat = (std::stoi(strs1[nodei])-1);
						material[elemmat] = 1;
					}
      			}
      		}
    	}
    }
    myMesh.elmaterial = material;
    myfile.close();

    // READ CURVED BOUNDARIES -- boundary flag = 1
    std::vector<int> BCs(node_X.size(),0);
	std::string keyword_BC = "*curve";

	myfile.open(filename);
	if (myfile.is_open())
	{
		// read in until you find the keyword *Boundary
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_BC);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> bci;bci.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						//The third side
						BCs[std::stoi(strs1[nodei])-1] = 1;
						//std::cout<<"BC_1 node: "<<strs1[nodei]<<"\n";
					}
      			}
      		}
    	}
    }
    myfile.close();

    // READ BOTTOM BOUNDARIES -- boundary flag = 2
	
	myfile.open(filename);
	keyword_BC = "*bottom";
	if (myfile.is_open())
	{
		// read in until you find the keyword *Boundary
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_BC);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> bci;bci.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						//The first side
						BCs[std::stoi(strs1[nodei])-1] = 2;
						//std::cout<<"BC_2 node: "<<strs1[nodei]<<"\n";
					}
      			}
      		}
    	}
    }
    myfile.close();

    // READ LEFT BOUNDARIES -- boundary flag = 3
	myfile.open(filename);
	keyword_BC = "*left";
	if (myfile.is_open())
	{
		// read in until you find the keyword *Boundary
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_BC);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> bci;bci.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						//The second side
						BCs[std::stoi(strs1[nodei])-1] = 3;
						//std::cout<<"BC_3 node: "<<strs1[nodei]<<"\n";
					}
      			}
      		}
    	}
    }
    myfile.close();

    // READ Corner -- boundary flag = 4
	myfile.open(filename);
	keyword_BC = "*corner";
	if (myfile.is_open())
	{
		// read in until you find the keyword *Boundary
		while ( getline (myfile,line) )
    	{
      		// check for the keyword
      		std::size_t found = line.find(keyword_BC);
			if (found!=std::string::npos)
      		{
      			// found the beginning of the nodes, so keep looping until you get '*'
      			while ( getline (myfile,line) )
      			{
      				if(line[0]=='*'){break;}
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> bci;bci.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=0;nodei<strs1.size();nodei++)
					{
						//The second side
						BCs[std::stoi(strs1[nodei])-1] = 4;
						//std::cout<<"BC_4 node: "<<strs1[nodei]<<"\n";
					}
      			}
      		}
    	}
    }
    myfile.close();

    myMesh.boundary_flag = BCs;
	
}