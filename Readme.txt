File tree:

./: 
    mesh_check.csv -- stores current mesh and BCs. Can be visulized through Visualization.nb
    PhibfuncGenerator.nb -- generate data files in PhibCurve/
    jobWoundCPP -- job file for job submission on Brown Cluster
    makefile -- must change the path of 'Eigen' and 'Boost' on different machine
        To compile: 
            make clean
            make apps/results)_circle_wound
    Visualization.nb -- visualize meshes, BCs generated by C++ code
    Adhesion code.nb -- Mathematica code for Figures 2-5

apps/: main executable file directory
    results_circle_wound -- compiled application

data/: output data directory **Must be created before running the compiled code
    fullwound*.vtk -- output data file, to be read by ParaView
    stress.csv -- output stress data file, format: fm, rho_i, s11, s22, s12(or stretch)

include/: head files directory
    myMeshGenerator.h 
    solver.h
    wound.h

meshing/: mesh files directory
    mymeshgenerator.cpp
        -- myRectangleMesh(...) -> generate Rectangular mesh 
        -- distanceX2E(...)
        -- conformMesh2Ellipse(...)
        -- readAbaqusToMymesh(...) -> read *.inp generated by ABAQUS. **Need manually assign BC nodes to each BCs' names in the code
    Sample.inp -- sample of modified .inp file. (currently read by the code)

objs/: objects file directory
    myMeshGenerator.o
    solver.o
    wound.o

PhibCurve/: data files of Phi_b vs lambda curves from different references
    PhibLamb_<FiberType><ReferenceName>f<ContractileForceValue>.txt -- <...> should be replaced by comments inside

src/: source code file directory
    solver.cpp
        -- fillDOFmap(...)
        -- evalElemJacobians(...)
        -- sparseWoundSolver(...) -> main solver
        -- readAbaqusInput(...) -x not used currently
        -- readTissue(...)
        -- writeTissue(...)
        -- writeParaview(...) -> write output
        -- writeNode(...)
        -- writeIP(...)
        -- writeElement(...)
    wound.cpp
        -- evalt_rho_rho(...) -> evaluate active traction wrt stretch
        -- evalDt_rho_rhoDEb(...) -> evaluate the first direvative of traction wrt stretch
        -- readPhib(...) -> read given phi_b vs lambda curve data from files
        -- LineInterpolation(...) -> give first-order interpolation
        -- evalWound(...) -> evaluate residuals and tangents
        -- evalFluxesSources(...)
        -- localWoundProblem(...) -> for local remodeling
        -- evalWoundFF(...) -x Not used currently, HGO paras works only for single material
        -- evalJacobian(...)
        -- LineQuadriIP(...)
        -- evalShapeFunctionsR(...)
        -- evalShapeFunctionsRxi(...)
        -- evalShapeFunctionsReta(...)
        -- evalPsif(...)
        -- evalSS(...) -x Not used currently, HGO paras works only for single material
        -- evalWoundRes(...) -x not used currently
    results_circle_wound.cpp
        -- frand(...)
        -- main(...) -> give parameters -> give BCs -> solve -> output

