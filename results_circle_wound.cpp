/*

RESULTS
circular wound problem.

Read a quad mesh defined by myself.
Then apply boundary conditions.
Solve.

*/

#include "wound.h"
#include "solver.h"
#include "myMeshGenerator.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept> 
#include <math.h> 
#include <string>
#include <time.h>

#include <Eigen/Dense> 
using namespace Eigen;

double frand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}


int main(int argc, char *argv[])
{

	std::cout<<"\nResults 3: full domain simulations \n";
	srand (time(NULL));

	bool debug_flag = 0; // 1 - debug mode on
	bool remodeling_off_flag = 1; // 1 - turn off remodeling

	//---------------------------------//
	// GLOBAL PARAMETERS
	//
	// for normalization
	double rho_phys = 10000.; // [cells/mm^3]
	double c_max = 1.0e-4; // [g/mm3] from tgf beta review, 5e-5g/mm3 was good for tissues
	//
	//double k0 = 0.0511; // neo hookean for skin, used previously, in MPa
	//double k0 = 0.0137791; // From fitting data of fibrin gel
	//double kf = 0.015; // stiffness of collagen in MPa, from previous paper
	//double k2 = 0.048; // nonlinear exponential coefficient, non-dimensional
	double t_rho = 0.0045/rho_phys; // force of fibroblasts in MPa, this is per cell. so, in an average sense this is the production by the natural density
	double t_rho_c = (0.045)/rho_phys; // force of myofibroblasts enhanced by chemical, I'm assuming normalized chemical, otherwise I'd have to add a normalizing constant
	double K_t_c = c_max/10.; // saturation of chemical on force. this can be calculated from steady state
	double D_rhorho = 0.0833; // diffusion of cells in [mm^2/hour], not normalized
	double D_rhoc = 1.66e-12/c_max/c_max; // diffusion of chemotactic gradient, an order of magnitude greater than random walk [mm^2/hour], not normalized
	double D_cc = 0.10; // diffusion of chemical TGF, not normalized. 
	double p_rho =0.034; // in 1/hour production of fibroblasts naturally, proliferation rate, not normalized, based on data of doubling rate from commercial use
	double p_rho_c = 0.034; // production enhanced by the chem, if the chemical is normalized, then suggest two fold,
	double p_rho_theta = 0.034; // enhanced production by theta
	double K_rho_c= c_max/10.; // saturation of cell proliferation by chemical, this one is definitely not crucial, just has to be small enough <cmax
	double d_rho = 0.2*p_rho ;// decay of cells, 20 percent of cells die per day
	double K_rho_rho = rho_phys; // saturation of cell by cell, from steady state
	double vartheta_e = 2.0; // physiological state of area stretch
	double gamma_theta = 5.; // sensitivity of heviside function
	double p_c_rho = 90.0e-16/rho_phys; // production of c by cells in g/cells/h
	double p_c_thetaE = 300.0e-16/rho_phys; // coupling of elastic and chemical, three fold
	double K_c_c = 1.; // saturation of chem by chem, from steady state
	double d_c = 0.01; // decay of chemical in 1/hours
	double f_m = 7.0; // contractile force in pN (5~20) for health tissue
	double rho_i = 500; // density of integrins for health tissue
	
	std::vector<std::string> fiber_ref_avail = {"CollagenSvensson","FibrinWLiu","CollagenSeliktar","FibrinGel","FibrinInterm002","FibrinInterm010","FibrinInterm050"};
	std::vector<double> kf_list = {0.004*17.599164683153674,0.028*3096.391865746513,0.0000815361,0.956401,0.00151989,0.00713618,0.0352455};
	std::vector<double> k2_list = {20.81811370546901,0.0005430957860466969,16.2405,0.0744698,20.5403,20.7689,20.8134};

	int ref_health = 0;
	int ref_wound = 1;

	//---------------------------------//
	std::string fiber_health = fiber_ref_avail[ref_health];
	std::string fiber_wound = fiber_ref_avail[ref_wound];

	double k0 = 0.00120111; // From fitting data of collagen gel (Seliktar)
	double kf = kf_list[ref_health];
	double k2 = k2_list[ref_health];
	double k0_wound = 0.0137791; // From fitting data of fibrin gel
	//double k0_wound = 0.0290923; //f_v = 0.02
	//double k0_wound = 0.0268981; //f_v = 0.1
	//double k0_wound = 0.0155923; // f_v = 0.5
	//double k0_wound = 0.00120111;
	double kf_wound = kf_list[ref_wound];
	double k2_wound = k2_list[ref_wound];


	// Pass parameters into myTissue
	std::vector<double> global_parameters = {k0,kf,k2,t_rho,t_rho_c,K_t_c,D_rhorho,D_rhoc,D_cc,p_rho,p_rho_c,p_rho_theta,K_rho_c,K_rho_rho,d_rho,vartheta_e,gamma_theta,p_c_rho,p_c_thetaE,K_c_c,d_c,f_m,rho_i,k0_wound,kf_wound,k2_wound};
	std::vector<std::string> global_para_names = {"k0","kf","k2","t_rho","t_rho_c","K_t_c","D_rhorho","D_rhoc","D_cc","p_rho","p_rho_c","p_rho_theta","K_rho_c","K_rho_rho","d_rho","vartheta_e","gamma_theta","p_c_rho","p_c_thetaE","K_c_c","d_c","f_m","rho_i","k0_wound","kf_wound","k2_wound"};


	//---------------------------------//
	// LOCAL PARAMETERS
	//
	// collagen fraction
	double p_phi = 0.002/rho_phys; // production by fibroblasts, natural rate in percent/hour, 5% per day
	double p_phi_c = p_phi; // production up-regulation, weighted by C and rho
	double p_phi_theta = p_phi; // mechanosensing upregulation. no need to normalize by Hmax since Hmax = 1
	double K_phi_c = 0.0001; // saturation of C effect on deposition. RANDOM?
	double d_phi = 0.000970; // rate of degradation, in the order of the wound process, 100 percent in one year for wound, means 0.000116 effective per hour means degradation = 0.002 - 0.000116
	double d_phi_rho_c = 0.5*0.000970/rho_phys/c_max;//0.000194; // degradation coupled to chemical and cell density to maintain phi equilibrium
	double K_phi_rho = rho_phys*p_phi/d_phi -1; // saturation of collagen fraction itself, from steady state
	//
	// fiber alignment
	double tau_omega = 10./(K_phi_rho+1); // time constant for angular reorientation, think 100 percent in one year
	//
	// dispersion parameter
	double tau_kappa = 1./(K_phi_rho+1); // time constant, on the order of a year
	double gamma_kappa = 5.; // exponent of the principal stretch ratio
	// 
	// permanent contracture/growth
	double tau_lamdaP_a = 1.0/(K_phi_rho+1); // time constant for direction a, on the order of a year
	double tau_lamdaP_s = 1.0/(K_phi_rho+1); // time constant for direction s, on the order of a year

	//
	std::vector<double> local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,tau_lamdaP_a,tau_lamdaP_s,gamma_theta,vartheta_e};
	std::vector<std::string> local_para_names = {"p_phi","p_phi_c","p_phi_theta","K_phi_c","K_phi_rho","d_phi","d_phi_rho_c","tau_omega","tau_kappa","gamma_kappa","tau_lamdaP_a","tau_lamdaP_s","gamma_theta","vartheta_e"};

	//
	// solution parameters
	double tol_local = 1e-5; // local tolerance
	int max_local_iter = 35; // max local iter
	//
	// other local stuff
	double PIE = 3.14159;
	//---------------------------------//
	
	
	//---------------------------------//
	// values for the wound
	double rho_wound = 0; // [cells/mm^3]
	double c_wound = 1.0e-4;
	//double phif0_wound=0.028;
	double phif0_wound=0.29;
	double kappa0_wound = 0.;
	double a0x = 1.0;//frand(0,1.);
	double a0y = 0.;//sqrt(1-a0x*a0x);
	Vector2d a0_wound;a0_wound<<a0x,a0y;
	Vector2d lamda0_wound;lamda0_wound<<1.,1.;
	Vector3d sigma0_wound;sigma0_wound<<0.,0.,0.;
	//---------------------------------//
	
	
	//---------------------------------//
	// values for the healthy
	double rho_healthy = 90; // [cells/mm^3]
	double c_healthy = 1.0e-5;
	//double phif0_healthy=0.004;
	double phif0_healthy=0.29;
	double kappa0_healthy = 1.;
	Vector2d a0_healthy;a0_healthy<<1.0,0.0;
	Vector2d lamda0_healthy;lamda0_healthy<<1.,1.;
	Vector3d sigma0_healthy;sigma0_healthy<<0.,0.,0.;
	//---------------------------------//
	
	// test
//	double rho_wound = 0; // [cells/mm^3]
//	double c_wound = 1.0e-5;
//	double phif0_wound=0.004;
//	double kappa0_wound = 0.1;
//	double a0x = 1;//frand(0,1.);
//	double a0y = 0;//sqrt(1-a0x*a0x);
//	Vector2d a0_wound;a0_wound<<a0x,a0y;
//	Vector2d lamda0_wound;lamda0_wound<<1.,1.;
//	Vector3d sigma0_wound;sigma0_wound<<0.,0.,0.;
//	double rho_healthy = 0; // [cells/mm^3]
//	double c_healthy = 1.0e-5;
//	double phif0_healthy=0.004;
//	double kappa0_healthy = 0.1;
//	Vector2d a0_healthy;a0_healthy<<1.0,0.0;
//	Vector2d lamda0_healthy;lamda0_healthy<<1.,1.;
//	Vector3d sigma0_healthy;sigma0_healthy<<0.,0.,0.;
	

	//---------------------------------//
	// Debugging module
	while(debug_flag==1){

		int debug_mode = 0;
		std::cout<<"Select part for debugging:\n";
		std::cout<<" 0 - Global parameters (default)\n";
		std::cout<<" 1 - Local parameters\n";
		std::cout<<" 2 - Other parameters\n";
		std::cout<<" 3 - Turn off remodeling\n";
		std::cout<<" 4 - Select outputs\n";
		std::cout<<"-1 - Done\n";
		std::cin>>debug_mode;

		char para_check = 'n';

		if(debug_mode==0){
			int para_id = -1; // set up default value
			// Show and change GLOBAL parameters
			std::cout<<"**** Global parameters debugging mode ****\n";
			std::cout<<"Reference:\n"<<fiber_health<<"\n";
			std::cout<<"Parameters:\n";
			std::cout<<-1<<"\tReference\t"<<fiber_health<<"\n";
			for(int name_i=0;name_i<global_para_names.size();name_i++){
				std::cout<<name_i<<"\t"<<global_para_names[name_i]<<"\t"<<global_parameters[name_i]<<"\n";
			}
			std::cout<<"Parameters and reference OK? y/n\n";
			std::cin>>para_check;
			while(para_check != 'y'){
				std::cout<<"Which one to change?\n";
				std::cin>>para_id;
				if(para_id==-1){
					std::cout<<"Available references:\n";
					for(int i_ref=0;i_ref<fiber_ref_avail.size();i_ref++){
						std::cout<<i_ref<<"\t"<<fiber_ref_avail[i_ref]<<"\n";
					}
					std::cout<<"Type reference id\n";
					std::cin>>ref_health;
					fiber_health = fiber_ref_avail[ref_health];
					global_parameters[1] = kf_list[ref_health];
					global_parameters[2] = k2_list[ref_health];
				}else{
					std::cout<<"Type values for "<<global_para_names[para_id]<<"\n";
					std::cin>>global_parameters[para_id];
				}
		
				std::cout<<"Reference:\n"<<fiber_health<<"\n";
				std::cout<<"Parameters:\n";
				std::cout<<-1<<"\tReference\t"<<fiber_health<<"\n";
				for(int name_i=0;name_i<global_para_names.size();name_i++){
					std::cout<<name_i<<"\t"<<global_para_names[name_i]<<"\t"<<global_parameters[name_i]<<"\n";
				}
				std::cout<<"All global parameters OK? y/n\n";
				std::cin>>para_check;
			}
		}

		if(debug_mode==1){
			int para_id = 0; // set up default value
			// Show and change LOCAL parameters
			std::cout<<"**** Local parameters debugging mode ****\n";
			std::cout<<"Parameters:\n";
			for(int name_i=0;name_i<local_para_names.size();name_i++){
				std::cout<<name_i<<"\t"<<local_para_names[name_i]<<"\t"<<global_parameters[name_i]<<"\n";
			}
			std::cout<<"Parameters and reference OK? y/n\n";
			std::cin>>para_check;
			while(para_check != 'y'){
				std::cout<<"Which one to change?\n";
				std::cin>>para_id;

				std::cout<<"Type values for "<<local_para_names[para_id]<<"\n";
				std::cin>>global_parameters[para_id];
		
				std::cout<<"Parameters:\n";
				for(int name_i=0;name_i<local_para_names.size();name_i++){
					std::cout<<name_i<<"\t"<<local_para_names[name_i]<<"\t"<<global_parameters[name_i]<<"\n";
				}
				std::cout<<"All global parameters OK? y/n\n";
				std::cin>>para_check;
			}
		}

		while(debug_mode==2){
			// Changing value of other parameters
			std::cout<<"\nAll parameers are OK? y/n";
			std::cin>>para_check;

			if(para_check=='y'){
				debug_mode=0;
			}
		}

		if(debug_mode==3){
			//Turn off remodeling
			std::cout<<"Remodeling now is ";
			if(remodeling_off_flag==1){
				std::cout<<"off.\n";
			}else{
				std::cout<<"on.\n";
			}

			std::cout<<"Type value to change remodeling status:\n"<<"0 - on\n"<<"1 - off\n";
			std::cin>>remodeling_off_flag;
			
		}
		
		if(debug_mode==4){
			//Choose outputs
			std::cout<<"Current outputs are:\n";
		}
		std::cout<<"All done? y/n\n";
		std::cin>>para_check;
		if(para_check=='y'){
			debug_flag = 0;
		}
	}

	if(remodeling_off_flag==1){
		d_phi = 0.; //turn off remodeling
		d_phi_rho_c = 0.; //turn off remodeling
		K_phi_rho =1.e16; //turn off remodeling
		tau_omega = 1.e16; //turn off remodeling
		tau_kappa = 1.e16; //turn off remodeling
		tau_lamdaP_a = 1.e16; //turn off remodeling
		tau_lamdaP_s = 1.e16; //turn off remodeling
		local_parameters = {p_phi,p_phi_c,p_phi_theta,K_phi_c,K_phi_rho,d_phi,d_phi_rho_c,tau_omega,tau_kappa,gamma_kappa,tau_lamdaP_a,tau_lamdaP_s,gamma_theta,vartheta_e};
	}


	//---------------------------------//
	// create mesh (only nodes and elements)
	std::cout<<"Going to create the mesh\n";
//	std::vector<double> rectangleDimensions = {0.0,100.0,0.0,100.0};
//	std::vector<int> meshResolution = {50,50};
//	QuadMesh myMesh = myRectangleMesh(rectangleDimensions, meshResolution);
	QuadMesh myMesh;
	std::string filename_abaqus_inp = "meshing/Sample.inp";
	readAbaqusToMymesh(filename_abaqus_inp.c_str(), myMesh);
	std::cout<<"Created the mesh with "<<myMesh.n_nodes<<" nodes and "<<myMesh.n_elements<<" elements\n";
	// print the mesh
	/*
	std::cout<<"nodes\n";
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		std::cout<<myMesh.nodes[nodei](0)<<","<<myMesh.nodes[nodei](1)<<"\n";
	}
	std::cout<<"elements\n";
	for(int elemi=0;elemi<myMesh.n_elements;elemi++){
		std::cout<<myMesh.elements[elemi][0]<<","<<myMesh.elements[elemi][1]<<","<<myMesh.elements[elemi][2]<<","<<myMesh.elements[elemi][3]<<"\n";
	}
	std::cout<<"boundary\n";
	for(int nodei=0;nodei<myMesh.n_nodes;nodei++){
		std::cout<<(nodei+1)<<":\t"<<myMesh.boundary_flag[nodei]<<"\n";
	}
	*/
	// create the other fields needed in the tissue struct.
	int n_elem = myMesh.n_elements;
	int n_node = myMesh.n_nodes;
	//
	// global fields rho and c initial conditions 
	std::vector<double> node_rho0(n_node,rho_healthy);
	std::vector<double> node_c0 (n_node,c_healthy);
	//
	// values at the integration points
	std::vector<double> ip_phi0(n_elem*4,phif0_healthy);
	std::vector<Vector2d> ip_a00(n_elem*4,a0_healthy);
	std::vector<double> ip_kappa0(n_elem*4,kappa0_healthy);
	std::vector<Vector2d> ip_lamda0(n_elem*4,lamda0_healthy);
	std::vector<Vector3d> ip_sigma0(n_elem*4,sigma0_healthy);
	std::vector<int> ip_fiber0(n_elem*4,0); // ip_fiber = 0 -- healthy, 1 -- wound
	std::vector<double> ip_rhoi0(n_elem*4,rho_i);  // integrin density at each ip
	std::vector<double> ip_fm0(n_elem*4,f_m);    // contractile force at each ip
	//
	// define ellipse
	double x_center = 0.;
	double y_center = 0.;
	double x_axis = 50.1;
	double y_axis = 50.1;
	double alpha_ellipse = 0.;
	// boundary conditions and definition of the wound
	double tol_boundary = 1e-5;
	std::map<int,double> eBC_x;
	std::map<int,double> eBC_rho;
	std::map<int,double> eBC_c;
	std::vector<int> eBC_moving;
	for(int nodei=0;nodei<n_node;nodei++){
		double x_coord = myMesh.nodes[nodei](0);
		double y_coord = myMesh.nodes[nodei](1);
		// check
//		if(myMesh.boundary_flag[nodei]==1){
			// insert the boundary condition for displacement
	//		std::cout<<"fixing node "<<nodei<<"\n";
//			eBC_x.insert ( std::pair<int,double>(nodei*2+0,myMesh.nodes[nodei](0)) ); // x coordinate
//			eBC_x.insert ( std::pair<int,double>(nodei*2+1,myMesh.nodes[nodei](1)) ); // y coordinate 
			// insert the boundary condition for rho
//			eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) ); 
			// insert the boundary condition for c
//			eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
//		}

		// only for quarter circle case
		// boundary flags:
		//     1 -- outer curve, x,y directions are free; rho and c are fixed
		//     2 -- bottom side, y direction, rho and c are fixed
		//     3 -- left side, x direction, rho and c are fixed
		//     4 -- bottom-left corner, x, y directions, rho and c are fixed
		if(myMesh.boundary_flag[nodei]==1){
			eBC_moving.push_back(nodei);
			eBC_x.insert ( std::pair<int,double>(nodei*2+0,1.005*myMesh.nodes[nodei](0)) ); // x coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*2+1,1.005*myMesh.nodes[nodei](1)) ); // y coordinate 
			// fix rho and c on curved side
			// insert the boundary condition for rho
			//eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) );
			eBC_rho.insert ( std::pair<int,double>(nodei,0) );
			// insert the boundary condition for c
			//eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
			eBC_c.insert   ( std::pair<int,double>(nodei,0) );
		}
	
		if(myMesh.boundary_flag[nodei]==2){
			// insert the Uy boundary condition for displacement
	//		std::cout<<"fixing node "<<nodei<<"\n";
//			eBC_x.insert ( std::pair<int,double>(nodei*2+0,1.02*myMesh.nodes[nodei](0)) ); // x coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*2+1,myMesh.nodes[nodei](1)) ); // y coordinate 
			//eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) ); 
			//eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
		}
		if(myMesh.boundary_flag[nodei]==3){
			// insert the Ux boundary condition for displacement
			eBC_x.insert ( std::pair<int,double>(nodei*2+0,myMesh.nodes[nodei](0)) ); // x coordinate
//			eBC_x.insert ( std::pair<int,double>(nodei*2+1,1.02*myMesh.nodes[nodei](1)) ); // y coordinate
//			eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) ); 
//			eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
		}
		if(myMesh.boundary_flag[nodei]==4){
			// insert the Uxy boundary condition for displacement
			eBC_x.insert ( std::pair<int,double>(nodei*2+0,myMesh.nodes[nodei](0)) ); // x coordinate
			eBC_x.insert ( std::pair<int,double>(nodei*2+1,myMesh.nodes[nodei](1)) ); // y coordinate 

//			eBC_rho.insert ( std::pair<int,double>(nodei,rho_healthy) ); 
//			eBC_c.insert   ( std::pair<int,double>(nodei,c_healthy) );
		}


		//fix rho and c at all nodes, currently

		
		// insert the boundary condition for rho
		eBC_rho.insert ( std::pair<int,double>(nodei,0) ); 
		// insert the boundary condition for c
		eBC_c.insert   ( std::pair<int,double>(nodei,0) );
//		eBC_x.insert ( std::pair<int,double>(nodei*2+0,myMesh.nodes[nodei](0)) );
//		eBC_x.insert ( std::pair<int,double>(nodei*2+1,myMesh.nodes[nodei](1)) );

		// check if it is in the center of the wound
//		double check_ellipse = pow((x_coord-x_center)*cos(alpha_ellipse)+(y_coord-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
//						pow((x_coord-x_center)*sin(alpha_ellipse)+(y_coord-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
//		if(check_ellipse<=1){
//			// inside
//			//std::cout<<"wound node "<<nodei<<"\n";
//			node_rho0[nodei] = rho_wound;
//			node_c0[nodei] = c_wound;
//		}
	}

	for(int elemi=0;elemi<n_elem;elemi++){
		std::vector<Vector3d> IP = LineQuadriIP();
		for(int ip=0;ip<IP.size();ip++)
		{
			double xi = IP[ip](0);
			double eta = IP[ip](1);
			// weight of the integration point
			double wip = IP[ip](2);
			std::vector<double> R = evalShapeFunctionsR(xi,eta);
			Vector2d X_IP;X_IP.setZero();
			for(int nodej=0;nodej<4;nodej++){
				X_IP += R[nodej]*myMesh.nodes[myMesh.elements[elemi][nodej]];
			}
			double check_ellipse_ip = pow((X_IP(0)-x_center)*cos(alpha_ellipse)+(X_IP(1)-y_center)*sin(alpha_ellipse),2)/(x_axis*x_axis) +\
						pow((X_IP(0)-x_center)*sin(alpha_ellipse)+(X_IP(1)-y_center)*cos(alpha_ellipse),2)/(y_axis*y_axis) ;
			if(check_ellipse_ip<=1.){
				//std::cout<<"IP node: "<<4*elemi+ip<<"\n";
				ip_phi0[elemi*4+ip] = phif0_wound;
				ip_a00[elemi*4+ip] = a0_wound;
				ip_kappa0[elemi*4+ip] = kappa0_wound;
				ip_lamda0[elemi*4+ip] = lamda0_wound;
				ip_sigma0[elemi*4+ip] = sigma0_wound;
				ip_fiber0[elemi*4+ip] = 1;
				//ip_rhoi0[elemi*4+ip]  = 500+3.6*(pow(X_IP(0),2)+pow(X_IP(1),2))-0.00072*pow(pow(X_IP(0),2)+pow(X_IP(1),2),2);
				//ip_rhoi0[elemi*4+ip]  = 500+11.88574*(pow(X_IP(0),2)+pow(X_IP(1),2))-0.00371429*pow(pow(X_IP(0),2)+pow(X_IP(1),2),2);
				//ip_rhoi0[elemi*4+ip]  = 10000-4*(pow(X_IP(0),2)+pow(X_IP(1),2))+0.0008*pow(pow(X_IP(0),2)+pow(X_IP(1),2),2);
				//ip_rhoi0[elemi*4+ip]  = 5000;
				//ip_fm0[elemi*4+ip] = 7;
				//if(check_ellipse_ip>=0.6){
				//	ip_fm0[elemi*4+ip] = 11;
				//}else{
				//	ip_fm0[elemi*4+ip] = 7;
				//}				
			}
			//if(myMesh.elmaterial[elemi]==1){
			//	//std::cout<<"IP node: "<<4*elemi+ip<<"\n";
			//	ip_phi0[elemi*4+ip] = phif0_wound;
			//	ip_a00[elemi*4+ip] = a0_wound;
			//	ip_kappa0[elemi*4+ip] = kappa0_wound;
			//	ip_lamda0[elemi*4+ip] = lamda0_wound;
			//	ip_sigma0[elemi*4+ip] = sigma0_wound;
			//	ip_fiber0[elemi*4+ip] = 1;
			//}
		}
	}
	// no neumann boundary conditions. 
	std::map<int,double> nBC_x;
	std::map<int,double> nBC_rho;
	std::map<int,double> nBC_c;
	
	// initialize my tissue
	tissue myTissue;
	// connectivity
	myTissue.LineQuadri = myMesh.elements;
	// parameters
	myTissue.fiber_health = fiber_health;
	myTissue.fiber_wound = fiber_wound;
	myTissue.global_parameters = global_parameters;
	myTissue.local_parameters = local_parameters;
	//
	myTissue.node_X = myMesh.nodes;
	myTissue.node_x0 = myMesh.nodes;
	myTissue.node_x = myMesh.nodes;
	myTissue.node_rho_0 = node_rho0;
	myTissue.node_rho = node_rho0;
	myTissue.node_c_0 = node_c0;
	myTissue.node_c = node_c0;
	myTissue.ip_phif_0 = ip_phi0;	
	myTissue.ip_phif = ip_phi0;	
	myTissue.ip_a0_0 = ip_a00;		
	myTissue.ip_a0 = ip_a00;	
	myTissue.ip_kappa_0 = ip_kappa0;	
	myTissue.ip_kappa = ip_kappa0;	
	myTissue.ip_lamdaP_0 = ip_lamda0;	
	myTissue.ip_lamdaP = ip_lamda0;	
	myTissue.ip_sigma = ip_sigma0;
	myTissue.ip_sigma_act = ip_sigma0; // only for initialization, not correct
	myTissue.ip_sigma_pas = ip_sigma0; // only for initialization, not correct
	myTissue.ip_fiber = ip_fiber0;
	myTissue.ip_rhoi = ip_rhoi0;
	myTissue.ip_fm = ip_fm0;
	//
	myTissue.eBC_x = eBC_x;
	myTissue.eBC_rho = eBC_rho;
	myTissue.eBC_c = eBC_c;
	myTissue.nBC_x = nBC_x;
	myTissue.nBC_rho = nBC_rho;
	myTissue.nBC_c = nBC_c;
	myTissue.eBC_moving = eBC_moving;
	//myTissue.time_final = 24*15;
	myTissue.time = 0.;
	myTissue.time_final = 1;
	myTissue.time_step = 0.1;
	myTissue.tol = 1e-8;
	myTissue.max_iter = 40;
	myTissue.n_node = myMesh.n_nodes;
	myTissue.n_quadri = myMesh.n_elements;
	myTissue.n_IP = 4*myMesh.n_elements;

	//Check BCs
//	std::cout<<"eBCs:\n";
//	std::cout<<"x,y\n";
//	for(auto it = eBC_x.cbegin(); it != eBC_x.cend(); ++it)
//	{
//    	std::cout << it->first << " " << it->second << "\n";
//	}
//	std::cout<<"rho\n";
//	for(auto it = eBC_rho.cbegin(); it != eBC_rho.cend(); ++it)
//	{
//    	std::cout << it->first << " " << it->second << "\n";
//	}
//	std::cout<<"c\n";
//	for(auto it = eBC_c.cbegin(); it != eBC_c.cend(); ++it)
//	{
//    	std::cout << it->first << " " << it->second << "\n";
//	}
	//write to checkink file
	std::ofstream savefile;
	savefile.open("mesh_check.csv", std::ofstream::trunc);
	savefile<<"Nodes: "<<"Node_i, x, y\n";
	for(int nodei=0;nodei<n_node;nodei++){
       	savefile <<nodei+1<<","<<myTissue.node_x[nodei](0)<<","<<myTissue.node_x[nodei](1)<<"\n";
	}
	savefile<<"Elements: "<<"Elem_i, Node_1, Node_2, Node_3, Node_4\n";
	for(int elemi=0;elemi<n_elem;elemi++){
		savefile<<elemi+1<<","<<myTissue.LineQuadri[elemi][0]<<","<<myTissue.LineQuadri[elemi][1]<<","<<myTissue.LineQuadri[elemi][2]<<","<<myTissue.LineQuadri[elemi][3]<<"\n";
	}
	savefile<<"Material: "<<"flag\n";
	for(int elmat=0;elmat<n_elem;elmat++){
       	savefile <<elmat+1<<","<<myMesh.elmaterial[elmat]<<"\n";
	}
	savefile<<"BCs\n";
	savefile<<"xy: Node(flattened), values\n";
	for(auto it = eBC_x.cbegin(); it != eBC_x.cend(); ++it)
	{
    	savefile << it->first << "," << it->second << "\n";
	}
	savefile<<"rho: Node, values\n";
	for(auto it = eBC_rho.cbegin(); it != eBC_rho.cend(); ++it)
	{
    	savefile << it->first << "," << it->second << "\n";
	}
	savefile<<"c: Node, values\n";
	for(auto it = eBC_c.cbegin(); it != eBC_c.cend(); ++it)
	{
    	savefile << it->first << "," << it->second << "\n";
	}
	savefile.close();

	//
	std::cout<<"filling dofs...\n";
	fillDOFmap(myTissue);
	std::cout<<"going to eval jacobians...\n";
	evalElemJacobians(myTissue);
	//
	//print out the Jacobians
//	std::cout<<"element jacobians\nJacobians= ";
//	std::cout<<myTissue.elem_jac_IP.size()<<"\n";
//	for(int i=0;i<myTissue.elem_jac_IP.size();i++){
//		std::cout<<"element: "<<i<<"\n";
//		for(int j=0;j<4;j++){
//			std::cout<<"ip; "<<i<<"\n"<<myTissue.elem_jac_IP[i][j]<<"\n";
//		}
//	}
	// print out the forward dof map
//	std::cout<<"Total :"<<myTissue.n_dof<<" dof\n";
//	for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
//		std::cout<<"x node*2+coord: "<<i<<", dof: "<<myTissue.dof_fwd_map_x[i]<<"\n";
//	}
//	for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
//		std::cout<<"rho node: "<<i<<", dof: "<<myTissue.dof_fwd_map_rho[i]<<"\n";
//	}
//	for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
//		std::cout<<"c node: "<<i<<", dof: "<<myTissue.dof_fwd_map_c[i]<<"\n";
//	}
	//
	// 
	std::cout<<"going to start solver\n";
	// save a node and an integration point to a file
	std::vector<int> save_node;save_node.clear();
	std::vector<int> save_ip;save_ip.clear();
	
		
	std::stringstream ss;
	std::string filename = "data/fullwound06"+ss.str()+"_";
	
	
	//----------------------------------------------------------//
	// SOLVE
	sparseWoundSolver(myTissue, filename, 1, save_node,save_ip);
	//----------------------------------------------------------//


	std::cout<<"References in use: "<<fiber_health<<","<<fiber_wound<<"\n";

	return 0;	
}
