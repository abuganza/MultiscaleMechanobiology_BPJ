/*
	Mesh generator
*/

#ifndef myMeshGenerator_h
#define myMeshGenerator_h

#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

// a simple structure to keep nodes and elements
struct QuadMesh
{
	int n_nodes;
	int n_elements;
	std::vector<Vector2d> nodes;
	std::vector<std::vector<int> > elements;
	std::vector<int> elmaterial;
	std::vector<int> boundary_flag; // for the nodes
};

// a simple quad mesh generation, really stupid one
QuadMesh myRectangleMesh(const std::vector<double> &rectangleDimensions, const std::vector<int> &meshResolution);

// conform the mesh
void conformMesh2Ellipse(QuadMesh &myMesh, std::vector<double> &ellipse);
double distanceX2E(std::vector<double> &ellipse, double x_coord, double y_coord,double mesh_size);

void readAbaqusToMymesh(const char* filename, QuadMesh &myMesh);

#endif