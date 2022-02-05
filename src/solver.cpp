/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/


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

//-------------------------------------------------------------------------------------//
// PRE PROCESSING
//-------------------------------------------------------------------------------------//


//---------------------------------------//
// FILL IN DOF MAPS
//---------------------------------------//
// NOTE: the structure has an empty dof map, but it already has the eBC, the essential
// boundary condition maps
void fillDOFmap(tissue &myTissue)
{
	// some mesh values
	int n_node = myTissue.node_X.size();
	int n_elem = myTissue.LineQuadri.size();
	
	// initialize the dof count and the maps
	int dof_count = 0;
	// displacements
	std::vector< int > dof_fwd_map_x(n_node*2,-1);
	// concentrations
	std::vector< int > dof_fwd_map_rho(n_node,-1);
	std::vector< int > dof_fwd_map_c(n_node,-1);
		
	// all dof inverse map
	std::vector< std::vector<int> > dof_inv_map;
	
	// loop over the node set
	for(int i=0;i<n_node;i++)
	{
		// check if node has essential boundary conditions for displacements
		for(int j=0; j<2; j++)
		{
			if(myTissue.eBC_x.find(i*2+j)==myTissue.eBC_x.end())
			{
				// no eBC_x, means this is a DOF
				// fill in forward map
				dof_fwd_map_x[i*2+j] = dof_count;
				// fill in inverse map
				std::vector<int> dofinvx = {0,i*2+j};
				dof_inv_map.push_back(dofinvx);
				dof_count+=1;
			}else{
				// this node is in fact in the eBC
				myTissue.node_x[i](j) = myTissue.eBC_x.find(i*2+j)->second;
			}
		}		
		if(myTissue.eBC_rho.find(i)==myTissue.eBC_rho.end())
		{
			dof_fwd_map_rho[i] = dof_count;
			std::vector<int> dofinvrho = {1,i};
			dof_inv_map.push_back(dofinvrho);
			dof_count+=1;
		}else{
			// this node is in fact in the eBC, 
			myTissue.node_rho[i] = myTissue.eBC_rho.find(i)->second;
		}
		if(myTissue.eBC_c.find(i)==myTissue.eBC_c.end())
		{
			dof_fwd_map_c[i] = dof_count;
			std::vector<int> dofinvc = {2,i};
			dof_inv_map.push_back(dofinvc);
			dof_count+=1;
		}else{
			// this node is in fact in the eBC, 
			myTissue.node_c[i] = myTissue.eBC_c.find(i)->second;
		}
	}
	myTissue.dof_fwd_map_x = dof_fwd_map_x;
	myTissue.dof_fwd_map_rho = dof_fwd_map_rho;
	myTissue.dof_fwd_map_c = dof_fwd_map_c;
	myTissue.dof_inv_map = dof_inv_map;
	myTissue.n_dof = dof_count;
}


//---------------------------------------//
// EVAL JACOBIANS
//---------------------------------------//

// NOTE: assume the mesh and boundary conditions have already been read. The following
// function stores the internal element variables, namely the Jacobians, maybe some other
// thing, but the Jacobians is the primary thing
//
// EVAL JACOBIANS
void evalElemJacobians(tissue &myTissue)
{
	// clear the vectors
	std::vector<std::vector<Matrix2d> > elem_jac_IP;
	// loop over the elements
	int n_elem = myTissue.LineQuadri.size();
	std::cout<<"evaluating element jacobians, over "<<n_elem<<" elements\n";
	for(int ei=0;ei<n_elem;ei++)
	{
		// this element connectivity
		std::vector<int> elem = myTissue.LineQuadri[ei];
		// nodal positions for this element
		std::vector<Vector2d> node_X_ni;
		for(int ni=0;ni<4;ni++)
		{
			node_X_ni.push_back(myTissue.node_X[elem[ni]]);
		}
		// compute the vector of jacobians
		std::vector<Matrix2d> jac_IPi = evalJacobian(node_X_ni);
		elem_jac_IP.push_back(jac_IPi);
	}
	// assign to the structure
	myTissue.elem_jac_IP = elem_jac_IP;
}


//-------------------------------------------------------------------------------------//
// SOLVER
//-------------------------------------------------------------------------------------//


//---------------------------------------//
// SPARSE SOLVER
//---------------------------------------//
//
// NOTE: at this point the struct is ready with all that is needed, including boundary and
// initial conditions. Also, the internal constants, namely the Jacobians, have been
// calculated and stored. Time to solve the global system
void sparseWoundSolver(tissue &myTissue, std::string filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip)
{
	int n_dof = myTissue.n_dof;
	std::cout<<"I will solve a small system of "<<n_dof<<" dof\n";
    VectorXd RR(n_dof);
	VectorXd SOL(n_dof);SOL.setZero();
    //SpMat KK(n_dof,n_dof); // sparse to solve with BiCG
    SparseMatrix<double, ColMajor> KK2(n_dof,n_dof);
	std::vector<T> KK_triplets;KK_triplets.clear();
	SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int> >   solver2;
	//std::cout<<"start parameters\n";
	// PARAMETERS FOR THE SIMULATION
	double time_final = myTissue.time_final;
	double time_step  = myTissue.time_step;
	double time = myTissue.time;
	double total_steps = (time_final-time)/time_step;
	
	// Save an original configuration file
	std::stringstream ss;
	ss << "REF";
	std::string filename_step = filename + ss.str()+".vtk";
	//std::cout<<"write paraview\n";
	writeParaview(myTissue,filename_step.c_str());
	//std::cout<<"declare variables for the solver\n";
	
	//------------------------------------//
	// DECLARE VARIABLES YOU'LL USE LATER
	// nodal positions for the element
	std::vector<Vector2d> node_x_ni;
	//			
	// concentration and cells for previous and current time
	std::vector<double> node_rho_0_ni;
	std::vector<double> node_c_0_ni;
	std::vector<double> node_rho_ni;
	std::vector<double> node_c_ni;
	//			
	// values of the structural variables at the IP
	std::vector<double> ip_phif_0_pi;
	std::vector<Vector2d> ip_a0_0_pi;
	std::vector<double> ip_kappa_0_pi;
	std::vector<Vector2d> ip_lamdaP_0_pi;
	std::vector<double> ip_phif_pi;
	std::vector<Vector2d> ip_a0_pi;
	std::vector<double> ip_kappa_pi;
	std::vector<Vector2d> ip_lamdaP_pi;
	std::vector<Vector3d> ip_sigma_pi;
	std::vector<Vector3d> ip_sigma_act_pi;
	std::vector<Vector3d> ip_sigma_pas_pi;
	std::vector<int> ip_fiber_pi;
	std::vector<double> ip_rhoi_pi;
	std::vector<double> ip_fm_pi;
	//
    // pieces of the Residuals
    VectorXd Re_x(8);
    VectorXd Re_rho(4); 
    VectorXd Re_c(4); 
	//
    // pieces of the Tangents
	MatrixXd Ke_x_x(8,8);
	MatrixXd Ke_x_rho(8,4);
	MatrixXd Ke_x_c(8,4);
	MatrixXd Ke_rho_x(4,8);
	MatrixXd Ke_rho_rho(4,4);
	MatrixXd Ke_rho_c(4,4);
	MatrixXd Ke_c_x(4,8);
	MatrixXd Ke_c_rho(4,4);
	MatrixXd Ke_c_c(4,4);
	//------------------------------------//
	
	// LOOP OVER TIME
	std::cout<<"start loop over time\n";
	for(int step=0;step<total_steps;step++)
	{
		// GLOBAL NEWTON-RAPHSON ITERATION
		int iter;
		double residuum;
		double residuum0;
		//std::cout<<"tissue tolerance: "<<myTissue.tol<<"\n";
		//std::cout<<"max iterations: "<<myTissue.max_iter<<"\n";

		// LOOP OVER PSEUDO LOADING TIME
		for(double t_load=1;t_load<=1;t_load=t_load+0.1){
			iter = 0;
			residuum  = 1.;
			residuum0 = 1.;
			while(residuum>myTissue.tol && iter<myTissue.max_iter)
			{
        	    // reset the solvers
        	    KK2.setZero();
        	    RR.setZero();
        	    KK_triplets.clear();
        	    SOL.setZero();
        	    
        	    // START LOOP OVER ELEMENTS
        	    int n_elem = myTissue.LineQuadri.size();
        	    for(int ei=0;ei<n_elem;ei++)
        	    {
        	    	// element stuff
        	    	
        	    	// connectivity of the linear elements
        	    	std::vector<int> elem_ei = myTissue.LineQuadri[ei];
        	    	//std::cout<<"ELEMENT "<<ei<<": "<<elem_ei[0]<<","<<elem_ei[1]<<","<<elem_ei[2]<<","<<elem_ei[3]<<"\n";
        	    	
					// nodal positions for this element
					node_x_ni.clear();
					
					// concentration and cells for previous and current time
					node_rho_0_ni.clear();
					node_c_0_ni.clear();
					node_rho_ni.clear();
					node_c_ni.clear();
					
					// values of the structural variables at the IP
					ip_phif_0_pi.clear();
					ip_a0_0_pi.clear();
					ip_kappa_0_pi.clear();
					ip_lamdaP_0_pi.clear();
					ip_phif_pi.clear();
					ip_a0_pi.clear();
					ip_kappa_pi.clear();
					ip_lamdaP_pi.clear();
					ip_sigma_pi.clear();
					ip_sigma_act_pi.clear();
					ip_sigma_pas_pi.clear();
					ip_fiber_pi.clear();
					ip_rhoi_pi.clear();
					ip_fm_pi.clear();
		
					for(int ni=0;ni<4;ni++){
						// deformed positions
						node_x_ni.push_back(myTissue.node_x[elem_ei[ni]]);
						//std::cout<<"pushing node "<<elem_ei[ni]<<": "<<node_x_ni[ni](0)<<","<<node_x_ni[ni](1)<<"\n";
						
						// cells and chemical
						node_rho_0_ni.push_back(myTissue.node_rho_0[elem_ei[ni]]);
						node_c_0_ni.push_back(myTissue.node_c_0[elem_ei[ni]]);
						node_rho_ni.push_back(myTissue.node_rho[elem_ei[ni]]);
						node_c_ni.push_back(myTissue.node_c[elem_ei[ni]]);
						
						// structural variables
						// conveniently inside this loop because n_ip = n_node
						ip_phif_0_pi.push_back(myTissue.ip_phif_0[ei*4+ni]);
						ip_phif_pi.push_back(myTissue.ip_phif[ei*4+ni]);
						ip_a0_0_pi.push_back(myTissue.ip_a0_0[ei*4+ni]);
						ip_a0_pi.push_back(myTissue.ip_a0[ei*4+ni]);
						ip_kappa_0_pi.push_back(myTissue.ip_kappa_0[ei*4+ni]);
						ip_kappa_pi.push_back(myTissue.ip_kappa[ei*4+ni]);
						ip_lamdaP_0_pi.push_back(myTissue.ip_lamdaP_0[ei*4+ni]);
						ip_lamdaP_pi.push_back(myTissue.ip_lamdaP[ei*4+ni]);
						ip_sigma_pi.push_back(myTissue.ip_sigma[ei*4+ni]);
						ip_sigma_act_pi.push_back(myTissue.ip_sigma_act[ei*4+ni]);
						ip_sigma_pas_pi.push_back(myTissue.ip_sigma_pas[ei*4+ni]);
						ip_fiber_pi.push_back(myTissue.ip_fiber[ei*4+ni]);
						ip_rhoi_pi.push_back(myTissue.ip_rhoi[ei*4+ni]);
						ip_fm_pi.push_back(myTissue.ip_fm[ei*4+ni]);
					}
					
            		// and calculate the element Re and Ke
            		// pieces of the Residuals
            		Re_x.setZero();
            		Re_rho.setZero(); 
            		Re_c.setZero(); 
	
            		// pieces of the Tangents
					Ke_x_x.setZero();
					Ke_x_rho.setZero();
					Ke_x_c.setZero();
					Ke_rho_x.setZero();
					Ke_rho_rho.setZero();
					Ke_rho_c.setZero();
					Ke_c_x.setZero();
					Ke_c_rho.setZero();
					Ke_c_c.setZero();
	
            		// subroutines to evaluate the element
            		//
            		//std::cout<<"going to eval wound\n";
            		evalWound(
            		time_step, 1,
            		myTissue.elem_jac_IP[ei],
            		myTissue.fiber_health, myTissue.fiber_wound, ip_fiber_pi,
            		myTissue.global_parameters,myTissue.local_parameters,
            		node_rho_0_ni,node_c_0_ni,
            		ip_phif_0_pi,ip_a0_0_pi,ip_kappa_0_pi,ip_lamdaP_0_pi,
            		node_rho_ni, node_c_ni,
            		ip_phif_pi,ip_a0_pi,ip_kappa_pi,ip_lamdaP_pi,
            		ip_rhoi_pi,ip_fm_pi,
            		ip_sigma_pi,ip_sigma_act_pi,ip_sigma_pas_pi,
            		node_x_ni,
            		Re_x, Ke_x_x, Ke_x_rho, Ke_x_c,
            		Re_rho, Ke_rho_x, Ke_rho_rho, Ke_rho_c,
            		Re_c, Ke_c_x, Ke_c_rho, Ke_c_c);
	
					//std::cout<<"Ke_x_x\n"<<Ke_x_x<<"\n";
					//std::cout<<"Ke_x_rho\n"<<Ke_x_rho<<"\n";
					//std::cout<<"Ke_x_c\n"<<Ke_x_c<<"\n";
					//std::cout<<"Ke_rho_x\n"<<Ke_rho_x<<"\n";
					//std::cout<<"Ke_rho_rho\n"<<Ke_rho_rho<<"\n";
					//std::cout<<"Ke_rho_c\n"<<Ke_rho_c<<"\n";
					//std::cout<<"Ke_c_x\n"<<Ke_c_x<<"\n";
					//std::cout<<"Ke_c_rho\n"<<Ke_c_rho<<"\n";
					//std::cout<<"Ke_c_c\n"<<Ke_c_c<<"\n";
					// store the new IP values
            		for(int ipi=0;ipi<4;ipi++){
            			myTissue.ip_phif[ei*4+ipi] = ip_phif_pi[ipi];
            			myTissue.ip_a0[ei*4+ipi] = ip_a0_pi[ipi];
            			myTissue.ip_kappa[ei*4+ipi] = ip_kappa_pi[ipi];
            			myTissue.ip_lamdaP[ei*4+ipi] = ip_lamdaP_pi[ipi];
            			myTissue.ip_sigma[ei*4+ipi] = ip_sigma_pi[ipi];
            			myTissue.ip_sigma_act[ei*4+ipi] = ip_sigma_act_pi[ipi];
            			myTissue.ip_sigma_pas[ei*4+ipi] = ip_sigma_pas_pi[ipi];
            		}
            		//std::cout<<"done with  wound\n";
            		// assemble into KK triplets array and RR
	
            		// LOOP OVER NODES
					for(int nodei=0;nodei<4;nodei++){
						// ASSEMBLE DISPLACEMENT RESIDUAL AND TANGENTS
						for(int coordi=0;coordi<2;coordi++){
							if(myTissue.dof_fwd_map_x[elem_ei[nodei]*2+coordi]>-1){
								// residual
								RR(myTissue.dof_fwd_map_x[elem_ei[nodei]*2+coordi]) += Re_x(nodei*2+coordi);
								// loop over displacement dof for the tangent
								for(int nodej=0;nodej<4;nodej++){
									for(int coordj=0;coordj<2;coordj++){
										if(myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj]>-1){
											T K_x_x_nici_njcj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*2+coordi],myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj],Ke_x_x(nodei*2+coordi,nodej*2+coordj)};
											KK_triplets.push_back(K_x_x_nici_njcj);
										}
									}
									// rho tangent
									if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
										T K_x_rho_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*2+coordi],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_x_rho(nodei*2+coordi,nodej)};
										KK_triplets.push_back(K_x_rho_nici_nj);
									}
									// C tangent
									if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
										T K_x_c_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*2+coordi],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_x_c(nodei*2+coordi,nodej)};
										KK_triplets.push_back(K_x_c_nici_nj);
									}
								}
							}
						}
						// ASSEMBLE RHO
						if(myTissue.dof_fwd_map_rho[elem_ei[nodei]]>-1){
							RR(myTissue.dof_fwd_map_rho[elem_ei[nodei]]) += Re_rho(nodei);
							// tangent of the rho
							for(int nodej=0;nodej<4;nodej++){
								for(int coordj=0;coordj<2;coordj++){
									if(myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj]>-1){
										T K_rho_x_ni_njcj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj],Ke_rho_x(nodei,nodej*2+coordj)};
										KK_triplets.push_back(K_rho_x_ni_njcj);
									}
								}
								if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
									T K_rho_rho_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_rho_rho(nodei,nodej)};
									KK_triplets.push_back(K_rho_rho_ni_nj);
								}
								if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
									T K_rho_c_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_rho_c(nodei,nodej)};
									KK_triplets.push_back(K_rho_c_ni_nj);
								}
							}
						}
						// ASSEMBLE C
						if(myTissue.dof_fwd_map_c[elem_ei[nodei]]>-1){
							RR(myTissue.dof_fwd_map_c[elem_ei[nodei]]) += Re_c(nodei);
							// tangent of the C
							for(int nodej=0;nodej<4;nodej++){
								for(int coordj=0;coordj<2;coordj++){
									if(myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj]>-1){
										T K_c_x_ni_njcj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*2+coordj],Ke_c_x(nodei,nodej*2+coordj)};
										KK_triplets.push_back(K_c_x_ni_njcj);
									}
								}
								if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
									T K_c_rho_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_c_rho(nodei,nodej)};
									KK_triplets.push_back(K_c_rho_ni_nj);
								}
								if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
									T K_c_c_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_c_c(nodei,nodej)};
									KK_triplets.push_back(K_c_c_ni_nj);
								}
							}
						}
					}
					// FINISH LOOP OVER NODES (for assembly)
				}
				// FINISH LOOP OVER ELEMENTS
				
				
				// residual norm
				double normRR = sqrt(RR.dot(RR));
				if(iter==0){
					//std::cout<<"first residual\n"<<RR<<"\nRe_rho\n"<<Re_rho<<"\nRe_c\n"<<Re_c<<"\n";
					residuum0 = normRR;
					if(residuum0<myTissue.tol){std::cout<<"no need to solve?: "<<residuum0<<"\n";break;}
					//std::cout<<"first tangents\nKe_x_x\n"<<Ke_x_x<<"\nKe_x_rho\n"<<Ke_x_rho<<"\nKe_x_c\n"<<Ke_x_c<<"\n";
					//std::cout<<"first tangents\nKe_rho_x\n"<<Ke_rho_x<<"\nKe_rho_rho\n"<<Ke_rho_rho<<"\nKe_rho_c\n"<<Ke_rho_c<<"\n";
					//std::cout<<"first tangents\nKe_c_x\n"<<Ke_c_x<<"\nKe_c_rho\n"<<Ke_c_rho<<"\nKe_c_c\n"<<Ke_c_c<<"\n";
				}
				else{residuum = normRR/(1+residuum0);}
				
				// SOLVE: one approach
				//std::cout<<"solve\n";
				//KK.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
				KK2.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
				KK2.makeCompressed();
				//std::cout<<"KK2\n"<<KK2<<"\n";
				//solver2.analyzePattern(KK2); 
				// Compute the numerical factorization 
				//solver2.factorize(KK2); 
				solver2.compute(KK2);
				//Use the factors to solve the linear system 
				SOL = solver2.solve(-1.*RR); 
				//std::cout<<"Sol done/n";
				// SOLVE: alternate
				//BiCGSTAB<SparseMatrix<double> > solver;
				//solver.compute(KK2);
				//VectorXd SOL = solver.solve(-RR);
				//std::cout << "#iterations:     " << solver.iterations() << std::endl;
  				//std::cout << "estimated error: " << solver.error()      << std::endl;
				//std::cout<<SOL2<<"\n";
				
				// update the solution
				double normSOL = sqrt(SOL.dot(SOL));
				for(int dofi=0;dofi<n_dof;dofi++)
				{
					std::vector<int> dof_inv_i = myTissue.dof_inv_map[dofi];
					if(dof_inv_i[0]==0){
						// displacement dof
						int nodei  = dof_inv_i[1]/2;
						int coordi = dof_inv_i[1]%2;
						myTissue.node_x[nodei](coordi)+=SOL(dofi);
					}else if(dof_inv_i[0]==1){
						// rho dof
						myTissue.node_rho[dof_inv_i[1]] += SOL(dofi);
					}else if(dof_inv_i[0]==2){
						// C dof
						myTissue.node_c[dof_inv_i[1]] += SOL(dofi);
					}
				}
				iter += 1;
				if(iter == myTissue.max_iter){std::cout<<"\nCheck, make sure residual is small enough\n";}
				//std::cout<<"End of iteration : "<<iter<<",\nResidual before increment: "<<normRR<<"\nIncrement norm: "<<normSOL<<"\n\n";
			}
			// FINISH WHILE LOOP OF NEWTON INCREMENTS
		}
		//FINISH LOOP OF PSEUDO LOADING TIME
		
		// ADVANCE IN TIME
		
		// nodal variables
		for(int nodei=0;nodei<myTissue.n_node;nodei++)
		{
			myTissue.node_rho_0[nodei] = myTissue.node_rho[nodei];
			myTissue.node_c_0[nodei] = myTissue.node_c[nodei] ;
		}
		// integration point variables
		for(int elemi=0;elemi<myTissue.n_quadri;elemi++)
		{
			for(int IPi=0;IPi<4;IPi++){
				myTissue.ip_phif_0[elemi*4+IPi] = myTissue.ip_phif[elemi*4+IPi];
				myTissue.ip_a0_0[elemi*4+IPi] = myTissue.ip_a0[elemi*4+IPi];
				myTissue.ip_kappa_0[elemi*4+IPi] = myTissue.ip_kappa[elemi*4+IPi];
				myTissue.ip_lamdaP_0[elemi*4+IPi] = myTissue.ip_lamdaP[elemi*4+IPi];
			}
		}
		
		time += time_step;
		std::cout<<"End of Newton increments, residual: "<<residuum<<"\nEnd of time step :"<<step<<", \nTime: "<<time<<"\n\n";
		
		// write out a paraview file.
		if(step%save_freq==0)	
		{
			std::stringstream ss;
			ss << step+1;
			std::string filename_step = filename + ss.str()+".vtk";
			std::string filename_step_tissue = filename + ss.str()+".txt";
			writeParaview(myTissue,filename_step.c_str());
			writeTissue(myTissue,filename_step_tissue.c_str(),time);
		}
		
		// write out node variables in a file
		for(int nodei=0;nodei<save_node.size();nodei++){
			std::stringstream ss;
			ss << save_node[nodei];
			std::string filename_nodei = filename +"node"+ ss.str()+".txt";
			if(step==0){
				std::ofstream savefile(filename_nodei.c_str());
				if (!savefile) {throw std::runtime_error("Unable to open output file.");}
				savefile<<"## SAVING NODE "<<save_node[nodei]<<"TIME X(0) X(1) RHO C\n";
				savefile.close();
			}
			writeNode(myTissue,filename_nodei.c_str(),save_node[nodei],time);
		}
		// write out iP variables in a file
		for(int ipi=0;ipi<save_ip.size();ipi++){
			std::stringstream ss;
			ss << save_ip[ipi];
			std::string filename_ipi = filename + "IP"+ss.str()+".txt";
			if(step==0){
				std::ofstream savefile(filename_ipi.c_str());
				if (!savefile) {throw std::runtime_error("Unable to open output file.");}
				savefile<<"## SAVING IP "<<save_node[ipi]<<"TIME phi a0(0) a0(1) kappa lamdaN(0) lamdaB(1)\n";
				savefile.close();
			}
			writeIP(myTissue,filename_ipi.c_str(),save_ip[ipi],time);
		}

		// Update moving boundaries
		for(int iBC=0;iBC<myTissue.eBC_moving.size();iBC++){
			int nodei = myTissue.eBC_moving[iBC];
			myTissue.node_X[nodei](0) += 0.005*myTissue.node_x0[nodei](0);
			myTissue.node_X[nodei](1) += 0.005*myTissue.node_x0[nodei](1);
			myTissue.node_x[nodei](0) += 0.005*myTissue.node_x0[nodei](0);
			myTissue.node_x[nodei](1) += 0.005*myTissue.node_x0[nodei](1);
		}
	}
	// FINISH TIME LOOP
}


//-------------------------------------------------------------------------------------//
// IO
//-------------------------------------------------------------------------------------//

//---------------------------------------//
// READ ABAQUS
//---------------------------------------//
//
// read in the Abaqus file and generate the mesh and fill in 
void readAbaqusInput(const char* filename,tissue &myTissue)
{
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
    myTissue.node_X = node_X;
    
	// READ ELEMENTS	
	std::vector<std::vector<int> > LineQuadri; LineQuadri.clear();
	myfile.open(filename);
	std::string keyword_element = "*Element";
	if (myfile.is_open())
	{
		// read in until you find the keyword *NODE
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
      				// the nodes for the C2D4 element
      				// also remember that abaqus has node numbering starting in 1
      				std::string line2;
      				getline (myfile,line2);
      				std::vector<std::string> strs1;
					boost::split(strs1,line,boost::is_any_of(","));
					std::vector<int> elemi;elemi.clear();
					//std::cout<<"line1: ";
					// CHECK //
					for(int nodei=1;nodei<strs1.size()-1;nodei++)
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
    myTissue.LineQuadri = LineQuadri;	
	
	// in addition to the connectivity and the nodes, some other things
	myTissue.n_node = node_X.size();
	myTissue.n_quadri = LineQuadri.size();
	
}

//---------------------------------------//
// READ MY OWN FILE
//---------------------------------------//
tissue readTissue(const char* filename)
{
	// initialize the structure
	tissue myTissue;
	
	std::ifstream myfile(filename);
	std::string line;
	if (myfile.is_open())
	{
		
		// time
		getline (myfile,line);
		std::stringstream ss0(line);
		ss0>>myTissue.time;
		
		// time final
		getline (myfile,line);
		std::stringstream ss1(line);
		ss1>>myTissue.time_final;
		
		// time step
		getline (myfile,line);
		std::stringstream ss2(line);
		ss2>>myTissue.time_step;
		
		// tol
		getline (myfile,line);
		std::stringstream ss3(line);
		ss3>>myTissue.tol;
		
		// max iter
		getline (myfile,line);
		std::stringstream ss4(line);
		ss4>>myTissue.max_iter;
		
		// global parameters
		int n_global_parameters;
		getline (myfile,line);
		std::stringstream ss5(line);
		ss5>>n_global_parameters;
		std::vector<double> global_parameters(n_global_parameters,0.);
		getline (myfile,line);
		std::stringstream ss6(line);
		for(int i=0;i<n_global_parameters;i++){
			ss6>>global_parameters[i];
		}
		myTissue.global_parameters = global_parameters;
		
		// local parameters
		int n_local_parameters;
		getline (myfile,line);
		std::stringstream ss7(line);
		ss7>>n_local_parameters;
		std::vector<double> local_parameters(n_local_parameters,0.);
		getline (myfile,line);
		std::stringstream ss8(line);
		for(int i=0;i<n_local_parameters;i++){
			ss8>>local_parameters[i];
		}
		myTissue.local_parameters = local_parameters;
		
		// n_node
		getline (myfile,line);
		std::stringstream ss9(line);
		ss9>>myTissue.n_node;
		
		// n_quadri
		getline (myfile,line);
		std::stringstream ss10(line);
		ss10>>myTissue.n_quadri;
		
		// n_IP
		getline (myfile,line);
		std::stringstream ss11(line);
		ss11>>myTissue.n_IP;
		if(myTissue.n_IP>4*myTissue.n_quadri || myTissue.n_IP<4*myTissue.n_quadri )
		{std::cout<<"number of integration points and elements don't match\n";myTissue.n_IP = 4*myTissue.n_quadri;}
		
		// n_dof
		getline (myfile,line);
		std::stringstream ss12(line);
		ss12>>myTissue.n_dof;
		
		// LineQuadri
		std::vector<int> temp_elem(4,0);
		std::vector<std::vector<int > > LineQuadri(myTissue.n_quadri,temp_elem);
		myTissue.LineQuadri = LineQuadri;
		for(int i=0;i<myTissue.LineQuadri.size();i++){
			getline (myfile,line);
			std::stringstream ss13(line);
			ss13>>myTissue.LineQuadri[i][0]; ss13>>myTissue.LineQuadri[i][1]; ss13>>myTissue.LineQuadri[i][2]; ss13>>myTissue.LineQuadri[i][3];
		}
		
		// boundaryNodes
		std::vector<int> boundaryNodes(myTissue.n_node,0);
		myTissue.boundaryNodes = boundaryNodes;
		for(int i=0;i<myTissue.boundaryNodes.size();i++){
			getline (myfile,line);
			std::stringstream ss14(line);
			ss14>>myTissue.boundaryNodes[i];
		}
		
		// node_X 
		std::vector<Vector2d> node_X(myTissue.n_node,Vector2d(0,0));
		myTissue.node_X = node_X;
		for(int i=0;i<myTissue.node_X.size();i++){
			getline (myfile,line);
			std::stringstream ss15(line);
			ss15>>myTissue.node_X[i](0);ss15>>myTissue.node_X[i](1);
		}
		
		// node_rho_0
		std::vector<double> node_rho_0(myTissue.n_node,0.0);
		myTissue.node_rho_0 = node_rho_0;
		for(int i=0;i<myTissue.node_rho_0.size();i++){
			getline (myfile,line);
			std::stringstream ss16(line);
			ss16>>myTissue.node_rho_0[i];
		}
		
		// node_c_0
		std::vector<double> node_c_0(myTissue.n_node,0.0);
		myTissue.node_c_0 = node_c_0;
		for(int i=0;i<myTissue.node_c_0.size();i++){
			getline (myfile,line);
			std::stringstream ss17(line);
			ss17>>myTissue.node_c_0[i];
		}
		
		// ip_phif_0
		std::vector<double> ip_phif_0(myTissue.n_IP,0.0);
		myTissue.ip_phif_0 = ip_phif_0;
		for(int i=0;i<myTissue.ip_phif_0.size();i++){
			getline (myfile,line);
			std::stringstream ss18(line);
			ss18>>myTissue.ip_phif_0[i];
		}
		
		// ip_a0_0
		std::vector<Vector2d> ip_a0_0(myTissue.n_IP,Vector2d(0,0));
		myTissue.ip_a0_0 = ip_a0_0;
		for(int i=0;i<myTissue.ip_a0_0.size();i++){
			getline (myfile,line);
			std::stringstream ss19(line);
			ss19>>myTissue.ip_a0_0[i](0);ss19>>myTissue.ip_a0_0[i](1);
		}
				
		// ip_kappa_0
		std::vector<double> ip_kappa_0(myTissue.n_IP,0.0);
		myTissue.ip_kappa_0 = ip_kappa_0;
		for(int i=0;i<myTissue.ip_kappa_0.size();i++){
			getline (myfile,line);
			std::stringstream ss20(line);
			ss20>>myTissue.ip_kappa_0[i];
		}
		
		// ip_lamdaP_0
		std::vector<Vector2d> ip_lamdaP_0(myTissue.n_IP,Vector2d(0,0));
		myTissue.ip_lamdaP_0 = ip_lamdaP_0;
		for(int i=0;i<myTissue.ip_lamdaP_0.size();i++){
			getline (myfile,line);
			std::stringstream ss21(line);
			ss21>>myTissue.ip_lamdaP_0[i](0);ss21>>myTissue.ip_lamdaP_0[i](1);
		}
		
		// node_x 
		std::vector<Vector2d> node_x(myTissue.n_node,Vector2d(0,0));
		myTissue.node_x = node_x;
		for(int i=0;i<myTissue.node_x.size();i++){
			getline (myfile,line);
			std::stringstream ss22(line);
			ss22>>myTissue.node_x[i](0);ss22>>myTissue.node_x[i](1);
		}
		
		// node_rho
		myTissue.node_rho = myTissue.node_rho_0;
		
		// node_c
		myTissue.node_c = myTissue.node_c_0;
		
		// ip_phif
		myTissue.ip_phif = myTissue.ip_phif_0;
		
		// ip_a0
		myTissue.ip_a0 = myTissue.ip_a0_0;
		
		// ip_kappa
		myTissue.ip_kappa = myTissue.ip_kappa_0;
		
		// ip_lamdaP
		myTissue.ip_lamdaP = myTissue.ip_lamdaP_0;
		
		// eBC_x
		int n_eBC_x;
		int dofx;
		double dofx_value;
		getline (myfile,line);
		std::stringstream ss23(line);
		ss23>>n_eBC_x;
		myTissue.eBC_x.clear();
		for(int i=0;i<n_eBC_x;i++){
			getline (myfile,line);
			std::stringstream ss24(line);
			ss24>>dofx;ss24>>dofx_value;
			myTissue.eBC_x.insert ( std::pair<int,double>(dofx,dofx_value) ); 
		}

		// eBC_rho
		int n_eBC_rho;
		int dofrho;
		double dofrho_value;
		getline (myfile,line);
		std::stringstream ss25(line);
		ss25>>n_eBC_rho;
		myTissue.eBC_rho.clear();
		for(int i=0;i<n_eBC_rho;i++){
			getline (myfile,line);
			std::stringstream ss26(line);
			ss26>>dofrho;ss26>>dofrho_value;
			myTissue.eBC_rho.insert ( std::pair<int,double>(dofrho,dofrho_value) ); 
		}

		// eBC_c
		int n_eBC_c;
		int dofc;
		double dofc_value;
		getline (myfile,line);
		std::stringstream ss27(line);
		ss27>>n_eBC_c;
		myTissue.eBC_c.clear();
		for(int i=0;i<n_eBC_c;i++){
			getline (myfile,line);
			std::stringstream ss28(line);
			ss28>>dofc;ss28>>dofc_value;
			myTissue.eBC_c.insert ( std::pair<int,double>(dofc,dofc_value) ); 
		}

		// nBC_x
		int n_nBC_x;
		double forcex_value;
		getline (myfile,line);
		std::stringstream ss29(line);
		ss29>>n_nBC_x;
		myTissue.nBC_x.clear();
		for(int i=0;i<n_nBC_x;i++){
			getline (myfile,line);
			std::stringstream ss30(line);
			ss30>>dofx;ss30>>forcex_value;
			myTissue.nBC_x.insert ( std::pair<int,double>(dofx,forcex_value) ); 
		}

		// nBC_rho
		int n_nBC_rho;
		double forcerho_value;
		getline (myfile,line);
		std::stringstream ss31(line);
		ss31>>n_nBC_rho;
		myTissue.nBC_rho.clear();
		for(int i=0;i<n_nBC_rho;i++){
			getline (myfile,line);
			std::stringstream ss32(line);
			ss32>>dofrho;ss32>>forcerho_value;
			myTissue.nBC_rho.insert ( std::pair<int,double>(dofrho,forcerho_value) ); 
		}

		// nBC_c
		int n_nBC_c;
		double forcec_value;
		getline (myfile,line);
		std::stringstream ss33(line);
		ss33>>n_nBC_c;
		myTissue.nBC_c.clear();
		for(int i=0;i<n_nBC_c;i++){
			getline (myfile,line);
			std::stringstream ss34(line);
			ss34>>dofc;ss34>>forcec_value;
			myTissue.nBC_c.insert ( std::pair<int,double>(dofc,forcec_value) ); 
		}

		// dof_fwd_map_x
		int n_dof_fwd_map_x;
		getline (myfile,line);
		std::stringstream ss35(line);
		ss35>>n_dof_fwd_map_x;
		std::vector<int> dof_fwd_map_x(n_dof_fwd_map_x,-1);
		myTissue.dof_fwd_map_x = dof_fwd_map_x;
		for(int i=0;myTissue.dof_fwd_map_x.size();i++){
			getline (myfile,line);
			std::stringstream ss36(line);
			ss36>>myTissue.dof_fwd_map_x[i];
		}
	
		// dof_fwd_map_rho
		int n_dof_fwd_map_rho;
		getline (myfile,line);
		std::stringstream ss37(line);
		ss37>>n_dof_fwd_map_rho;
		std::vector<int> dof_fwd_map_rho(n_dof_fwd_map_rho,-1);
		myTissue.dof_fwd_map_rho = dof_fwd_map_rho;
		for(int i=0;myTissue.dof_fwd_map_rho.size();i++){
			getline (myfile,line);
			std::stringstream ss38(line);
			ss38>>myTissue.dof_fwd_map_rho[i];
		}
		
		// dof_fwd_map_c
		int n_dof_fwd_map_c;
		getline (myfile,line);
		std::stringstream ss39(line);
		ss39>>n_dof_fwd_map_c;
		std::vector<int> dof_fwd_map_c(n_dof_fwd_map_c,-1);
		myTissue.dof_fwd_map_c = dof_fwd_map_c;
		for(int i=0;myTissue.dof_fwd_map_c.size();i++){
			getline (myfile,line);
			std::stringstream ss40(line);
			ss40>>myTissue.dof_fwd_map_c[i];
		}
	
		// dof_inv_map
		int n_dof_inv_map;
		getline (myfile,line);
		std::stringstream ss41(line);
		ss41>>n_dof_inv_map;
		std::vector<int> temp_inv_dof(2,0);
		std::vector<std::vector<int> > dof_inv_map(n_dof_inv_map,temp_inv_dof);
		myTissue.dof_inv_map = dof_inv_map;
		for(int i=0;myTissue.dof_inv_map.size();i++){
			getline (myfile,line);
			std::stringstream ss42(line);
			ss42>>myTissue.dof_inv_map[i][0];ss42>>myTissue.dof_inv_map[i][1];
		}		
	}
	myfile.close();
	evalElemJacobians(myTissue);
	return myTissue;
}


//---------------------------------------//
// WRITE OUT MY OWN FILE
//---------------------------------------//
//
void writeTissue(tissue &myTissue, const char* filename,double time)
{
	std::ofstream savefile(filename);
	if (!savefile) {
		throw std::runtime_error("Unable to open output file.");
	}
	savefile<<time<<"\n";
	savefile<<myTissue.time_final<<"\n";
	savefile<<myTissue.time_step<<"\n";
	savefile<<myTissue.tol<<"\n";
	savefile<<myTissue.max_iter<<"\n";	
	savefile<<myTissue.global_parameters.size()<<"\n";
	for(int i=0;i<myTissue.global_parameters.size();i++){
		savefile<<myTissue.global_parameters[i]<<" ";
	}
	savefile<<"\n";
	savefile<<myTissue.local_parameters.size()<<"\n";
	for(int i=0;i<myTissue.local_parameters.size();i++){
		savefile<<myTissue.local_parameters[i]<<" ";
	}
	savefile<<"\n";
	savefile<<myTissue.n_node<<"\n";
	savefile<<myTissue.n_quadri<<"\n";
	savefile<<myTissue.n_IP<<"\n";
	savefile<<myTissue.n_dof<<"\n";
	for(int i=0;i<myTissue.LineQuadri.size();i++){
		savefile<<myTissue.LineQuadri[i][0]<<" "<<myTissue.LineQuadri[i][1]<<" "<<myTissue.LineQuadri[i][2]<<" "<<myTissue.LineQuadri[i][3]<<"\n";
	}
	for(int i=0;i<myTissue.boundaryNodes.size();i++){
		savefile<<myTissue.boundaryNodes[i]<<"\n";
	}
	for(int i=0;i<myTissue.node_X.size();i++){
		savefile<<myTissue.node_X[i](0)<<" "<<myTissue.node_X[i](1)<<"\n";
	}
	for(int i=0;i<myTissue.node_rho_0.size();i++){
		savefile<<myTissue.node_rho_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.node_c_0.size();i++){
		savefile<<myTissue.node_c_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.ip_phif_0.size();i++){
		savefile<<myTissue.ip_phif_0[i]<<"\n";
	}
	for(int i=0;i<myTissue.ip_a0_0.size();i++){
		savefile<<myTissue.ip_a0_0[i](0)<<" "<<myTissue.ip_a0_0[i](1)<<"\n";
	}
	for(int i=0;i<myTissue.ip_kappa_0.size();i++){
		savefile<<myTissue.ip_kappa_0[i]<<"\n";
	}	
	for(int i=0;i<myTissue.ip_lamdaP_0.size();i++){
		savefile<<myTissue.ip_lamdaP_0[i](0)<<" "<<myTissue.ip_lamdaP_0[i](1)<<"\n";
	}
	for(int i=0;i<myTissue.node_x.size();i++){
		savefile<<myTissue.node_x[i](0)<<" "<<myTissue.node_x[i](1)<<"\n";
	}
	std::map<int,double>::iterator it_map_BC;
	savefile<<myTissue.eBC_x.size()<<"\n";
	for(it_map_BC = myTissue.eBC_x.begin(); it_map_BC != myTissue.eBC_x.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.eBC_rho.size()<<"\n";
	for(it_map_BC = myTissue.eBC_rho.begin(); it_map_BC != myTissue.eBC_rho.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.eBC_c.size()<<"\n";
	for(it_map_BC = myTissue.eBC_c.begin(); it_map_BC != myTissue.eBC_c.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_x.size()<<"\n";
	for(it_map_BC = myTissue.nBC_x.begin(); it_map_BC != myTissue.nBC_x.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_rho.size()<<"\n";
	for(it_map_BC = myTissue.nBC_rho.begin(); it_map_BC != myTissue.nBC_rho.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.nBC_c.size()<<"\n";
	for(it_map_BC = myTissue.nBC_c.begin(); it_map_BC != myTissue.nBC_c.end(); it_map_BC++) {
    	// iterator->first = key
    	// iterator->second = value
		savefile<<it_map_BC->first<<" "<<it_map_BC->second<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_x.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_x.size();i++){
		savefile<<myTissue.dof_fwd_map_x[i]<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_rho.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_rho.size();i++){
		savefile<<myTissue.dof_fwd_map_rho[i]<<"\n";
	}
	savefile<<myTissue.dof_fwd_map_c.size()<<"\n";
	for(int i=0;i<myTissue.dof_fwd_map_c.size();i++){
		savefile<<myTissue.dof_fwd_map_c[i]<<"\n";
	}
	savefile<<myTissue.dof_inv_map.size()<<"\n";
	for(int i=0;i<myTissue.dof_inv_map.size();i++){
		savefile<<myTissue.dof_inv_map[i][0]<<" "<<myTissue.dof_inv_map[i][1]<<"\n";
	}
	savefile.close();
}

//---------------------------------------//
// WRITE OUT A PARAVIEW FILE
//---------------------------------------//
void writeParaview(tissue &myTissue, const char* filename)
{
	std::ofstream savefile(filename);
	if (!savefile) {
		throw std::runtime_error("Unable to open output file.");
	}
	savefile<<"# vtk DataFile Version 2.0\nCartilage\nASCII\nDATASET UNSTRUCTURED_GRID\n";
	savefile<<"POINTS "<<myTissue.node_x.size()<<" double\n";
	for(int i=0;i<myTissue.node_x.size();i++)
	{
		savefile<<myTissue.node_x[i](0)<<" "<<myTissue.node_x[i](1)<<" 0.0\n";
	}
	savefile<<"CELLS "<<myTissue.LineQuadri.size()<<" "<<myTissue.LineQuadri.size()*5<<"\n";
	for(int i=0;i<myTissue.LineQuadri.size();i++)
	{
		savefile<<"4";
		for(int j=0;j<4;j++)
		{
			savefile<<" "<<myTissue.LineQuadri[i][j];
		}
		savefile<<"\n";
	}
	savefile<<"CELL_TYPES "<<myTissue.LineQuadri.size()<<"\n";
	for(int i=0;i<myTissue.LineQuadri.size();i++)
	{
		savefile<<"9\n";
	}
	
	// SAVE ATTRIBUTES
	// up to four scalars I can plot...
	// first bring back from the integration points to the nodes
	std::vector<double> node_phi(myTissue.n_node,0);
	std::vector<Vector2d> node_a0(myTissue.n_node,Vector2d(0,0));
	std::vector<double> node_kappa(myTissue.n_node,0);
	std::vector<Vector2d> node_lamdaP(myTissue.n_node,Vector2d(0,0));
	std::vector<Vector3d> node_sigma(myTissue.n_node,Vector3d(0,0,0));
	std::vector<Vector3d> node_sigma_act(myTissue.n_node,Vector3d(0,0,0));
	std::vector<Vector3d> node_sigma_pas(myTissue.n_node,Vector3d(0,0,0));
	std::vector<int> node_ip_count(myTissue.n_node,0);
	//std::cout<<"saving attributes in paraview file\n";
	for(int elemi=0;elemi<myTissue.n_quadri;elemi++){
		for(int ip=0;ip<4;ip++){
			node_phi[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_phif[elemi*4+ip];
			node_a0[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_a0[elemi*4+ip];
			node_kappa[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_kappa[elemi*4+ip];
			node_lamdaP[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_lamdaP[elemi*4+ip];
			node_sigma[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_sigma[elemi*4+ip];
			node_sigma_act[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_sigma_act[elemi*4+ip];
			node_sigma_pas[myTissue.LineQuadri[elemi][ip]]+=myTissue.ip_sigma_pas[elemi*4+ip];
			node_ip_count[myTissue.LineQuadri[elemi][ip]] += 1;
		}
	}
	for(int nodei=0;nodei<myTissue.n_node;nodei++){
		node_phi[nodei] = node_phi[nodei]/node_ip_count[nodei];
		node_a0[nodei] = node_a0[nodei]/node_ip_count[nodei];
		node_kappa[nodei] = node_kappa[nodei]/node_ip_count[nodei];
		node_lamdaP[nodei] = node_lamdaP[nodei]/node_ip_count[nodei];
		node_sigma[nodei] = node_sigma[nodei]/node_ip_count[nodei];
		node_sigma_act[nodei] = node_sigma_act[nodei]/node_ip_count[nodei];
		node_sigma_pas[nodei] = node_sigma_pas[nodei]/node_ip_count[nodei];
	}
	// rho, c, phi, theta
	savefile<<"POINT_DATA "<<myTissue.n_node<<"\nSCALARS rho_c double "<<4<<"\nLOOKUP_TABLE default\n";
	for(int i=0;i<myTissue.n_node;i++){
		// For mac Paraview
		if(myTissue.node_c[i]<1e-10 && myTissue.node_c[i]>-1e-10){
			myTissue.node_c[i] = 0.0;
		}
		if(node_phi[i]<1e-10 && node_phi[i]>-1e-10){
			node_phi[i] = 0.0;
		}
		if(node_lamdaP[i](0)<1e-10 && node_lamdaP[i](0)>-1e-10){
			node_lamdaP[i](0) = 0.0;
		}
		if(node_lamdaP[i](1)<1e-10 && node_lamdaP[i](1)>-1e-10){
			node_lamdaP[i](1) = 0.0;
		}
		savefile<<myTissue.node_rho[i]<<" "<<myTissue.node_c[i]<<" "<<node_phi[i]<<" "<<node_lamdaP[i](0)*node_lamdaP[i](1)<<"\n";
	}
	// write out the total stresses in the current configuration
	savefile<<"SCALARS sigma double "<<3<<"\nLOOKUP_TABLE default\n";
	for(int i=0;i<myTissue.n_node;i++){
		// For mac Paraview
		for(int jj=0;jj<3;jj++){
			if(node_sigma[i](jj)<1e-10 && node_sigma[i](jj)>-1e-10){
				node_sigma[i](jj) = 0.0;
			}
		}
		savefile<<node_sigma[i](0)<<" "<<node_sigma[i](1)<<" "<<node_sigma[i](2)<<"\n";
	}
	// write out the active stresses in the current configuration
	savefile<<"SCALARS sigma_act double "<<3<<"\nLOOKUP_TABLE default\n";
	for(int i=0;i<myTissue.n_node;i++){
		// For mac Paraview
		for(int jj=0;jj<3;jj++){
			if(node_sigma_act[i](jj)<1e-10 && node_sigma_act[i](jj)>-1e-10){
				node_sigma_act[i](jj) = 0.0;
			}
		}
		savefile<<node_sigma_act[i](0)<<" "<<node_sigma_act[i](1)<<" "<<node_sigma_act[i](2)<<"\n";
	}
	// write out the passive stresses in the current configuration
	savefile<<"SCALARS sigma_pas double "<<3<<"\nLOOKUP_TABLE default\n";
	for(int i=0;i<myTissue.n_node;i++){
		// For mac Paraview
		for(int jj=0;jj<3;jj++){
			if(node_sigma_pas[i](jj)<1e-10 && node_sigma_pas[i](jj)>-1e-10){
				node_sigma_pas[i](jj) = 0.0;
			}
		}
		savefile<<node_sigma_pas[i](0)<<" "<<node_sigma_pas[i](1)<<" "<<node_sigma_pas[i](2)<<"\n";
	}
	// write out the displacements
	savefile<<"VECTORS disp double\n";
	for(int i=0;i<myTissue.n_node;i++){
		double ux, uy;
		ux = myTissue.node_x[i](0)-myTissue.node_x0[i](0);
		uy = myTissue.node_x[i](1)-myTissue.node_x0[i](1);
		savefile<<ux<<" "<<uy<<" 0\n";
	}

	// write out the fiber direction
	savefile<<"VECTORS a0 double\n";
	for(int i=0;i<myTissue.n_node;i++){
		savefile<<node_a0[i](0)<<" "<<node_a0[i](1)<<" 0\n";
	}

	savefile.close();
}

//---------------------------------------//
// write NODE data to a file
//---------------------------------------//
void writeNode(tissue &myTissue,const char* filename,int nodei,double time)
{
	// write node i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	savefile<< time<<","<<myTissue.node_x[nodei](0)<<","<<myTissue.node_x[nodei](1)<<","<<myTissue.node_rho[nodei]<<","<<myTissue.node_c[nodei]<<"\n";
	savefile.close();
}

//---------------------------------------//
// write Integration Point data to a file
//---------------------------------------//
void writeIP(tissue &myTissue,const char* filename,int ipi,double time)
{
	// write integration point i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	savefile<< time<<","<<myTissue.ip_phif[ipi]<<","<<myTissue.ip_a0[ipi](0)<<","<<myTissue.ip_a0[ipi](1)<<","<<myTissue.ip_kappa[ipi]<<","<<myTissue.ip_lamdaP[ipi](0)<<","<<myTissue.ip_lamdaP[ipi](1)<<"\n";
	savefile.close();
}

//---------------------------------------//
// write Element data to a file
//---------------------------------------//
void writeElement(tissue &myTissue,const char* filename,int elemi,double time)
{
	// write element i to a file
	std::ofstream savefile;
	savefile.open(filename, std::ios_base::app);
	// average the nodes and the integration points of this element
	std::vector<int> element = myTissue.LineQuadri[elemi];
	double rho=0;
	double c = 0;
	Vector2d x;x.setZero();
	double phif = 0;
	Vector2d a0;a0.setZero();
	double kappa = 0;
	Vector2d lamdaP;lamdaP.setZero();
	for(int i=0;i<4;i++){
		x += myTissue.node_x[element[i]];
		rho += myTissue.node_rho[element[i]];
		c += myTissue.node_c[element[i]];
		phif += myTissue.ip_phif[elemi*4+i];
		a0 += myTissue.ip_a0[elemi*4+i];
		kappa += myTissue.ip_kappa[elemi*4+i];
		lamdaP += myTissue.ip_lamdaP[elemi*4+i];
	}
	x = x/4.; rho = rho/4.; c = c/4.;
	phif = phif/4.; a0 = a0/4.; kappa = kappa/4.; lamdaP = lamdaP/4.;
	savefile<<time<<","<<phif<<","<<a0(0)<<","<<a0(1)<<","<<kappa<<","<<lamdaP(0)<<","<<lamdaP(1)<<","<<rho<<","<<c<<"\n";
}