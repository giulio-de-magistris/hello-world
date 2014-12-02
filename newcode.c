#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <string.h>
#include <malloc.h>

/*---Size of the lattice---*/
#define Lx 200
#define Ly 200
#define Lz 1

#define TWODIMENSIONS 1

int domain_volume;
int LyLz;

/*---Number of timesteps--*/
#define Nmax 500000
#define stepskip 500
#define cpstep 25000

/*---Number of cells---*/
#define nphase 180
#define starting_phases 160

/*---Constants---*/
#define Pi 3.141592653589793
#define Pihalf 1.570796326794897
#define onesixth (1./6.)
#define onetwelth (1./12.)

/*---------------------------*/
/*---Simulation Parameters---*/
/*---------------------------*/

/*---Timestep---*/
#define dt 1

/*---Binary fluid parameters---*/
#define alpha2phi 0.5 /* was 0.25*/
#define Kphi 0.28 /* was 0.14 */
#define phi_0 2.
#define M 0.1 /*mobility*/

/*---Leslie-Ericksen parameter---*/
#define alpha2 0.0
#define K 0.04

#define alpha4 0.01 /* was 0.01*/ /* FAVORS ISOTROPIC PHASE IN POLARISATION */
#define beta 0.05 /* anchoring */

#define G 1. /*rotational diffusion constant*/

/*---Cell interaction parameters---*/
#define epsilon 0.01
#define delta 0.1
/*---Thresholds---*/
#define repulsion_threshold 0.1
#define adhesion_threshold 0.1
#define volume_threshold 1. /*was 1.*/
#define growth_threshold_low 0.1
#define growth_threshold_high 2.1

/*---Active parameters---*/
#define w1 0.0075 /*self advection*/
#define growthrate 0.000001
#define volume_equil 500
#define volume_division 400
#define spacing_division 0

/*-------------------------------------*/
/*---Density and polarisation fields---*/
/*-------------------------------------*/

/*---Fields---*/
double *phi;
double *phiold;

double *phitotal;

double *Px, *Py, *Pz;
double *Pxold, *Pyold, *Pzold;

/*---Chemical potential and molecular field---*/
double *mu;
double *h_phi;
double *hx, *hy, *hz;
/*--- or ---*/
/*
typedef struct{
	double phase[nphase];

} density;
typedef struct{
	double px[nphase];
	double py[nphase];
	double pz[nphase];
} polarisation;
*/
/*---Auxiliary fields----*/
double *deltaphiPx, *deltaphiPy, *deltaphiPz;
double *laplacianphi;
double *laplacianphitotal;


/*-----------------*/
/*---Observables---*/
/*-----------------*/
double *Rx, *Ry, *Rz;
double *Vx, *Vy, *Vz;
double *XXcm, *YYcm, *ZZcm;
double *XYcm, *XZcm, *YZcm;

double *eigen_axis1;
double *eigen_direction1x, *eigen_direction2x;
double *eigen_direction1y, *eigen_direction2y;
double* eigen_direction1_modulus;

double *total_mass;
double *volume;
double *average_density;

/*---Multiphase---*/
int phase_counter;
int config_space;

/*---Wound Parameters---*/
int initial_wound_size;
int empty_counter;

/*----------------------------*/
/*---Functions declarations---*/
/*----------------------------*/

/*---Initialization functions---*/
void memory_allocation(void);
void memory_allocation_lattice(void);
void memory_allocation_fields(void);
void memory_allocation_auxiliary_fields(void);
void memory_allocation_observables(void);
void initializeneighbours();
void initializefields();

/*---Initial conditions---*/
void initialize_from_file();
void initialize_from_mc();
void initialize_squaredroplet();
void initialize_rounddroplet();
void initialize_removal();
int check_center_of_mass(int l, double removal_distance);
void initialize_emulsion();
void initialize_HCP();

/*---Observables function---*/
void total_phi();
void cell_mass_0(int l);
void cell_mass(int l);
void cell_position(int l);
void cell_velocity(int l);
void cell_inertial_axes(int l);
void cells_distance(int l, int m);

int wound_size(int n);


/*---Chemical potential and molecular field---*/

void phi_function(int l);
void P_function(int l);

void chemical_potential(int l);
void streaming_and_diffusion(int l);

/*---Growth & division---*/
void growth(int l);
void division(int l, int n);

/*---Updates---*/
void old_fields(int l);
void auxiliary_fields(int l);
void updatephi0(int l);
void updateP0(int l);

/*---Plot Functions---*/
void plots(int n);

void plot_fields(int n);
void plot_wound_size(int n);
void plot_position(int n);
void plot_velocity(int n);
void checkpoint(int n);

void print_to_screen(int n);
void plot_laplacians(int n);

void plot_initial_config(void);
void plot_final_config(int n);

/*---Mathematical functions---*/
double max(double, double);
double min(double, double);
/*---Derivatives---*/
double dx_n(double *phi, int idx);
double dy_n(double *phi, int idx);
double dz_n(double *phi, int idx);
double laplacian_n(double *field, int idx);
double upwind_x(double vx, double *phi, int idx);
double upwind_y(double vy, double *phi, int idx);
double upwind_z(double vz, double *phi, int idx);

/*---Neighbours list---*/
int *idxf, *idxb, *idxr, *idxl, *idxu, *idxd;
/*
int *idxfr, *idxfl, *idxbr, *idxbl, *idxfu, *idxfd, *idxbu, *idxbd;
int *idxru, *idxrd, *idxlu, *idxld;
*/
/*---Switches---*/
#define READ_FROM_FILE 0
#define READ_FROM_MC 1
#define SQUARE_DROPLET 0
#define ROUND_DROPLET 0
#define REMOVAL 1
#define EMULSION 0
#define HCP 0

#define PLOT_FIELDS 1
#define PLOT_POSITION 1
#define PLOT_VELOCITY 0

void twodimensional(void );

/*---Timers---*/
clock_t t, totaltime;

int main(int argc, char** argv){
	int n;
	int l, temp_phase_counter;
	
	n=0;
	
	memory_allocation();
	initializeneighbours();
	initializefields();
	
	for( l=0; l < starting_phases ; l++ ){
		cell_mass(l);
		#if PLOT_POSITION
			cell_position(l);
		#endif
		#if PLOT_VELOCITY
			cell_velocity(l);
		#endif
	}
	
	plots(0);
	#if TWODIMENSIONS
	twodimensional();
	#endif
	plot_initial_config();
	totaltime = clock();
	t = clock();
	for( n = 1 ; n < Nmax + 1 ; n++ ){
		//t = clock();
		#pragma omp parallel
		{
			#pragma omp for nowait
			for( l = 0 ; l < temp_phase_counter ; l++ ){
				//printf("igetpasthere\n");
				old_fields(l); //saves configuration from previous timestep
				auxiliary_fields(l); //computes derivatives to be used in the increments of phi & P
				
				phi_function(l); //computes increment of the density fields
				growth(l);
				
				P_function(l); //computes increment of the polarisation fields
				
				updatephi0(l); //sums the increment to phi's
				updateP0(l); //sums the increment to P's
				
				cell_mass(l); //computes cell mass.
				#if PLOT_POSITION
				cell_position(l);
				#endif
				#if PLOT_VELOCITY
				cell_velocity(l);
				#endif
				
				if(volume[l] > volume_division ){
					division(l, n);//cell l is divided, phase_counter is increased, new cell is created
				}
			}
			#pragma omp barrier
			{
				#pragma omp single
				{
					total_phi();//computes the total density field everywhere
					n = wound_size(n);					
				}
			}
		}
		//t = clock() - t ;
		//printf("Single iteration time %.1E\n", ((float) t )/CLOCKS_PER_SEC);
		temp_phase_counter = phase_counter;
		#if TWODIMENSIONS
		twodimensional();
		#endif		
		if( n%stepskip == 0 ) {
			plots(n);
			t = clock() - t;
			printf("%d iterations time %.2E\n", stepskip, ((float) t )/CLOCKS_PER_SEC);
		}
		
		if(n%cpstep == 0){
			checkpoint(n);
		}
	}
	totaltime = clock() - totaltime ;
	printf("\n\nTotal time elapsed = %f seconds\n", ((float)totaltime)/CLOCKS_PER_SEC);
}

void memory_allocation(){
	phase_counter = starting_phases;
	config_space = phase_counter * Lx * Ly * Lz;
	domain_volume = Lx * Ly * Lz;
	LyLz = Ly * Lz;

	memory_allocation_lattice();
	memory_allocation_fields();
	memory_allocation_auxiliary_fields();
	memory_allocation_observables();
}

void memory_allocation_lattice(void ){ 
	idxf = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
	idxb = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
	idxr = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
	idxl = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
	idxu = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
	idxd = ( int *) calloc((Lx*Ly*Lz*nphase), sizeof(int));
}

void memory_allocation_fields(void ){
	phi = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Px = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Py = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Pz = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	
	phiold = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Pxold = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Pyold = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	Pzold = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	
	phitotal = (double *) calloc((Lx*Ly*Lz), sizeof(double)); 
	
	mu = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	h_phi= (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	
	hx = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	hy = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	hz = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	
}

void memory_allocation_auxiliary_fields(void){
	deltaphiPx = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	deltaphiPy = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	deltaphiPz = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	
	laplacianphi = (double *) calloc((Lx*Ly*Lz*nphase), sizeof(double));
	laplacianphitotal = (double *) calloc((Lx*Ly*Lz), sizeof(double));
}

void memory_allocation_observables(void){
	Rx = (double *) calloc(nphase, sizeof(double));
	Ry = (double *) calloc(nphase, sizeof(double));
	Rz = (double *) calloc(nphase, sizeof(double));
	
	Vx = (double *) calloc(nphase, sizeof(double));
	Vy = (double *) calloc(nphase, sizeof(double));
	Vz = (double *) calloc(nphase, sizeof(double));
	
	XXcm = (double *) calloc(nphase, sizeof(double));
	YYcm = (double *) calloc(nphase, sizeof(double));
	ZZcm = (double *) calloc(nphase, sizeof(double));
	
	XYcm = (double *) calloc(nphase, sizeof(double));
	XZcm = (double *) calloc(nphase, sizeof(double));
	YZcm = (double *) calloc(nphase, sizeof(double));
	
	eigen_axis1 = (double *) calloc(nphase, sizeof(double));
	eigen_direction1x = (double *) calloc(nphase, sizeof(double));
	eigen_direction1y = (double *) calloc(nphase, sizeof(double));
	eigen_direction1_modulus = (double *) calloc(nphase, sizeof(double));
	eigen_direction2x = (double *) calloc(nphase, sizeof(double));
	eigen_direction2y = (double *) calloc(nphase, sizeof(double));
	
	total_mass = (double *) calloc(nphase, sizeof(double));
	average_density = (double *) calloc(nphase, sizeof(double));
	volume = (double *) calloc(nphase, sizeof(double));
}

void initializeneighbours(){
	int idx, idxt;
	int x,y,z,l;
	
	for (idxt = 0 ; idxt < domain_volume ; idxt++ ) {
		for(l = 0 ; l < nphase ; l++ ) {
			idx = idxt + l * domain_volume;
			idxf[idx] = idx + LyLz;
			idxb[idx] = idx - LyLz;
			idxr[idx] = idx + Lz;
			idxl[idx] = idx - Lz;
			idxu[idx] = idx + 1;
			idxd[idx] = idx - 1;
		}
	}
	/*---Periodic Boundary Conditions---*/
	for( l = 0 ; l  < nphase ; l++){
		for(x = 0 ; x < Lx ; x++ ){
			for( y = 0 ; y < Ly ; y++ ){
				idx = l * domain_volume + x * LyLz + y * Lz;
				idxd[idx] = idx + (Lz-1);
				idxu[idx + (Lz-1)] = idx;
			}
		}
	}
	for( l = 0 ; l  < nphase ; l++){
		for(y = 0 ; y < Ly ;  y++){
			for(z = 0 ; z < Lz ; z++){
				idx = l * domain_volume + y * Lz + z ;
				idxb[idx] = idx + (Lx-1) * LyLz;
				idxf[idx + (Lx-1) * LyLz] = idx;
			}
		}
	}
	for( l = 0 ; l  < nphase ; l++){
		for(x = 0 ; x < Lx ; x++){
			for(z = 0 ; z < Lz ; z++){
				idx = l * domain_volume + x * LyLz + z;
				idxl[idx] = idx + (Ly-1) * Lz;
				idxr[idx + (Ly-1) * Lz] = idx;
			}
		}
	}
}


void initializefields(void){
	int idx, maxidx;
	
	maxidx = domain_volume * nphase;
	for(idx = 0 ; idx < maxidx ; idx++) {
		phiold[idx] = 0.;
		phi[idx] = 0.;
		Px[idx] = 0.;
		Py[idx] = 0.;
		Pz[idx] = 0.;
	}
	
	#if READ_FROM_FILE
	initialize_from_file();
	#endif
	
	#if READ_FROM_MC
	initialize_from_mc();
	#endif
	
	#if SQUARE_DROPLET
	initialize_squaredroplet();
	#endif
	
	#if ROUND_DROPLET
	initialize_rounddroplet();
	#endif
	
	#if HCP
	initialize_HCP();
	#endif
	
	#if EMULSION
	initialize_emulsion();
	#endif
	
	#if REMOVAL
	initialize_removal();
	#endif
	
	total_phi();
	wound_size(0);
	initial_wound_size = empty_counter;
}

void initialize_from_file(void){
	int x,y,z,l;
	int dummyx, dummyy, dummyz;
	int idx;
	FILE *input;
	
	input = fopen("starting.dat","r");
	for( x = 0 ; x < Lx ; x++) {
		for( y = 0; y < Ly; y++ ) {
			for( z = 0 ; z < Lz ; z++ ) {
				fscanf(input, " %d %d %d", &dummyx, &dummyy, &dummyz);			
				for(l = 0; l < starting_phases ; l++){
					idx = l * domain_volume + x * LyLz + y * Lz + z;
					fscanf(input, " %lf %lf %lf %lf", &Px[idx], &Py[idx], &Pz[idx], &phi[idx]);
				}
			}
		}
	}
	fclose(input);
}

void initialize_from_mc(void){
	int x,y,z,l;
	int dummyx, dummyy, dummyz;
	int idx;
	FILE *input;
	
	input = fopen("starting.dat","r");
	for( x = 0 ; x < Lx ; x++) {
		for( y = 0; y < Ly; y++ ) {
			for( z = 0 ; z < Lz ; z++ ) {
				fscanf(input, "%d", &l);
				idx = (l-1) * domain_volume + x * LyLz + y * Lz + z;
				phi[idx] = 2.;
			}
		}
	}
	fclose(input);
}

void initialize_squaredroplet(void){
	int x,y,z;
	int idx1;
	int size = 14;
	
	for( x = (Lx-size)/2 ; x < (Lx+size)/2 ; x++ ) {
		for( y = (Ly-size)/2 ; y < (Ly+size)/2 ; y++) {
			for( z = 0 ; z < Lz ; z++ ) {
				idx1 = x * LyLz + y * Lz + z;
				phi[idx1] = 2.0;
				Px[idx1] = 0.; 
				Py[idx1] = 0.; 
				Pz[idx1] = 0.; 
			}
		}		
	}
}

void initialize_rounddroplet(void){
	int x,y,z;
	int idx1;
	double radius = 10;
	double r;
	
	for( x = 0 ; x < Lx ; x++ ) {
		for( y = 0 ; y < Ly ; y++) {
			for( z = 0 ; z < Lz ; z++ ) {
				r = sqrt( (x - Lx/2)*(x - Lx/2) + (y - Ly/2)*(y - Ly/2) );
				idx1 = x * LyLz + y * Lz + z;
				phi[idx1] = 0. + 1. * ( 1. + tanh(radius - r) );
				Px[idx1] = 0.; 
				Py[idx1] = 0.; 
				Pz[idx1] = 0.; 
			}
		}
	}
}

void initialize_HCP(void){
	int x,y,z,l;
	int idx;
	int xidx, yidx;
	double diameter = 10;
	double R1;
	double halfdiameter;
	int nhor = 4, nver = 4;
	double spacing = 1.;
	
	halfdiameter = 0.5 * diameter;
        for ( l=0; l < starting_phases; l++ ){
		xidx = Lx/3 + floor( ( l%(nhor) + 0.25 * ( 1 + pow(-1, (floor( l/nhor )) - 1 ) ) )*(diameter+spacing) );
		yidx = Ly/3 + floor( floor(l/nver) * (diameter+spacing) * 0.5*sqrt(3) );
		
		for ( x = 0 ; x < Lx  ; x++ ){
			for ( y = 0 ; y < Ly ; y++){
				for ( z = 0 ; z < Lz ; z++ ){
					R1 = sqrt( (x-xidx) * (x - xidx) + (y-yidx) * (y-yidx) );
					idx = l * domain_volume + x * LyLz + y * Lz + z;
					phi[idx] = (1. + tanh( halfdiameter - R1));
				}
			}   
		}
	}
}

void initialize_emulsion(void){
}

void initialize_removal(void){
	int x,y,z,m,l;
	int idx, idxnext;
	int final_cells;
	int count;
	
	double distance;
	double removal_distance = 80;
	total_phi();
	
	for( l = 0 ; l < phase_counter ; l++){
		cell_mass(l);
		cell_position(l);
	}
	
	final_cells = starting_phases;
	count = 0;
	printf("Removing cells from tissue...\n");
	for(l=0; l < final_cells; l++){
		distance = sqrt( (Rx[l] - Lx/2 ) * (Rx[l]- Lx/2)  + ( Ry[l]- Ly/2 ) * (Ry[l]- Ly/2 ) + ( Rz[l]- Lz/2 ) * ( Rz[l]- Lz/2 ));
		//printf("%d %lf\n",l,  distance);
		if( (check_center_of_mass(l, removal_distance) > 0) && (distance < removal_distance)){
			count++;
			final_cells--;
			for( m = l ; m < final_cells+1 ; m++){
				for(x = 0 ; x < Lx ; x++ ){
					for( y = 0 ; y < Ly ; y++ ){
						for( z = 0 ; z < Lz ; z++){
							idx = m * domain_volume + x * LyLz + y * Lz + z;
							idxnext = (m+1) * domain_volume + x * LyLz + y * Lz + z;
							Px[idx] =  Px[idxnext];
							Py[idx] =  Py[idxnext];
							Pz[idx] =  Pz[idxnext];
							phi[idx] = phi[idxnext];
						}
					}
				}
			}
			
			for(m = 0 ; m < phase_counter ; m++ ){
				cell_mass(m);
				cell_position(m);
			}
			l--;
		}
	}
	for( m = final_cells ; m < starting_phases ; m++){
		for(x = 0 ; x < Lx ; x++ ){
			for( y = 0 ; y< Ly ; y++ ){
				for( z = 0 ; z < Lz ; z++){
					idx = m * domain_volume + x * LyLz + y * Lz + z;
					
					Px[idx] = 0.;
					Py[idx] = 0.;
					Pz[idx] = 0.;
					phi[idx] = 0.;
				} 
			}
		}
	}
	phase_counter -= count; 
}

int check_center_of_mass(int l, double removal_distance){
	int x, y, z;
	int idx;
	double distance;
	int output;
	output = 0;
	for(x = 0 ; x < Lx ; x++ ){
		for(y = 0 ; y < Ly ; y++){
			for(z = 0 ; z < Lz ; z++){
				distance = sqrt((x - Lx/2) * (x - Lx/2) + (y - Ly/2) * (y - Ly/2) + (z - Lz/2) * (z - Lz/2));
				if(distance < removal_distance){
					idx = l * domain_volume + x * LyLz + y * Lz + z;
					if(phi[idx] > volume_threshold ){
						output = 1;
						x = Lx;
						y = Ly;
						z = Lz;
					}
				}
			}
		}
	}
	return(output);
}

void cell_mass(int l){
	int idx;
	int minidx, maxidx;
	
	total_mass[l] = 0;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		if(phi[idx] > volume_threshold){
			total_mass[l] += phi[idx];
		}
	}
	volume[l] = 0.5 * total_mass[l];
	average_density[l] = total_mass[l] / domain_volume;
}
/*
void cell_mass(int l){
	int idx;
	int minidx, maxidx;
	
	total_mass[l] = 0;
	volume[l] = 0; 
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		total_mass[l] += phi[idx];
		volume[l] += 0.25 * phi[idx] * phi[idx] * (3. - phi[idx]);
	}
	average_density[l] = total_mass[l] / domain_volume;
}
*/

void cell_position(int l){
	int idx;
	int idxt;
	int x,y,z;
	int minidx, maxidx;
	
	Rx[l] = 0;
	Ry[l] = 0;
	Rz[l] = 0;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		idxt = idx % domain_volume;
		z = (idxt % LyLz ) % Lz ; 
		y = floor( ( idxt % LyLz ) / Lz ) ;
		x = floor( idxt / LyLz );
		
		if(phi[idx] > volume_threshold){
			Rx[l] += phi[idx] * x;
			Ry[l] += phi[idx] * y;
			Rz[l] += phi[idx] * z;
		}
	}
	
	Rx[l] /= total_mass[l];
	Ry[l] /= total_mass[l];
	Rz[l] /= total_mass[l];	
	
}

void cell_velocity(int l){
	int idx;
	int idxt;
	int x,y,z;
	int minidx, maxidx;
	
	Vx[l] = 0;
	Vy[l] = 0;
	Vz[l] = 0;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		idxt = idx % domain_volume;
		z = (idxt % LyLz ) % Lz ; 
		y = floor( ( idxt % LyLz ) / Lz ) ;
		x = floor( idxt / LyLz );
		
		if(phi[idx] > volume_threshold){
			Vx[l] += w1 * phi[idx] * Px[idx];
			Vy[l] += w1 * phi[idx] * Py[idx];
			Vz[l] += w1 * phi[idx] * Pz[idx];
		}
	}
	
	Vx[l] /= total_mass[l];
	Vy[l] /= total_mass[l];
	Vz[l] /= total_mass[l];	
	
}

void cell_inertial_axes(int l){
	int idx, idxt;
	int x,y,z;
	
	int minidx, maxidx;
	
	XXcm[l] = 0;
	YYcm[l] = 0;
	ZZcm[l] = 0;
	
	XYcm[l] = 0;
	XZcm[l] = 0;
	YZcm[l] = 0;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		idxt = idx % domain_volume;
		z = (idxt % LyLz ) % Lz ; 
		y = floor( ( idxt % LyLz ) / Lz ) ;
		x = floor( idxt / LyLz );
		
		if(phi[idx] > volume_threshold){
			XXcm[l] += phi[idx] * x * x;
			YYcm[l] += phi[idx] * y * y;
			ZZcm[l] += phi[idx] * z * z;
			
			XYcm[l] += phi[idx] * x * y;
			XZcm[l] += phi[idx] * x * z;
			YZcm[l] += phi[idx] * y * z;
		}
	}
	
	XXcm[l] -= Rx[l] * Rx[l] * total_mass[l];
	YYcm[l] -= Ry[l] * Ry[l] * total_mass[l];
	ZZcm[l] -= Rz[l] * Rz[l] * total_mass[l];
	
	XYcm[l] -= Rx[l] * Ry[l] * total_mass[l];
	XZcm[l] -= Ry[l] * Rz[l] * total_mass[l];
	YZcm[l] -= Ry[l] * Rz[l] * total_mass[l];
	
	eigen_axis1[l] = 0.5 * ( XXcm[l] + YYcm[l] + sqrt( ( XXcm[l] - YYcm[l] ) * ( XXcm[l] - YYcm[l] ) + 4. * XYcm[l]*XYcm[l] ) );
	
	eigen_direction1x[l] = XXcm[l] - eigen_axis1[l];
	eigen_direction1y[l] = - XYcm[l];
	
	eigen_direction1_modulus[l] = sqrt(eigen_direction1x[l] * eigen_direction1x[l] + eigen_direction1y[l] * eigen_direction1y[l]);
	
	eigen_direction1x[l] /= eigen_direction1_modulus[l];
	eigen_direction1y[l] /= eigen_direction1_modulus[l];

	eigen_direction2x[l] = eigen_direction1x[l];
	eigen_direction2y[l] = - eigen_direction1y[l];
}

void cells_distance(int l, int m){
}

int wound_size(int n){
	int l, idxt, idx;
	int empty_check;
	int output;
	
	empty_counter = 0;
	
	for(idxt = 0 ; idxt < domain_volume ; idxt++){
		empty_check = 0;
		for(l = 0 ; l < phase_counter ; l++ ) {
			idx = l * domain_volume + idxt; 
			if(phi[idx] > volume_threshold/2){
				empty_check++;
			}
		}
		if(empty_check == 0){
			empty_counter++;
		}
	}
	
	if(empty_counter < initial_wound_size/1000){
		plot_final_config(n);
		output = Nmax;		
	}else{
		output = n;
	}
	return(output);
}


void total_phi(){
	int idx, idxt;
	int l;
	for( idxt = 0 ; idxt < domain_volume ; idxt++ ) {
		phitotal[idxt] = 0. ;
		for( l = 0 ; l < phase_counter ; l++ ) {
			idx =  l * domain_volume + idxt;
			phitotal[idxt] += phi[idx];
		}
	}
	
	for(idxt = 0 ; idxt < domain_volume ; idxt++) {
		laplacianphitotal[idxt] = laplacian_n(phitotal,idxt);
	}
	
}

void old_fields(int l){
	int idx;
	int minidx, maxidx;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for( idx = minidx ; idx < maxidx ; idx++){
		phiold[idx] = phi[idx];
		Pxold[idx] = Px[idx];
		Pyold[idx] = Py[idx];
		Pzold[idx] = Pz[idx];
	}
}

void auxiliary_fields(int l){
	int idx; 
	int minidx, maxidx;
	double deltaphi;
	
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	
	for( idx = minidx ; idx < maxidx ; idx++ ) {
		deltaphi = phi[idx] * phi[idx] * ( phi[idx] - phi_0 ) * ( phi[idx] - phi_0 );
		deltaphiPx[idx] = deltaphi * Px[idx];
		deltaphiPy[idx] = deltaphi * Py[idx];
		deltaphiPz[idx] = deltaphi * Pz[idx];
		
		laplacianphi[idx] = laplacian_n(phi,idx);
	}
	
}


void phi_function(int l){
	int idx, idxt;
	
	double ddeltaphi;
	double Pgradphitot;
	double divdeltaphiP;
	double vx, vy, vz;
	double vxdphidx, vydphidy, vzdphidz;
	double divP;
	
	int minidx, maxidx;
	
	for ( idxt = 0 ; idxt < domain_volume ; idxt++ ){
		
		idx = l * domain_volume + idxt;
		
		ddeltaphi = phi[idx] * (  phi[idx] * ( phi[idx] - 3.0 ) + 2.0);
		Pgradphitot = Px[idx] * dx_n(phitotal, idxt) + Py[idx] * dy_n(phitotal, idxt) + Pz[idx] * dz_n(phitotal, idxt);
		divdeltaphiP = dx_n(deltaphiPx, idx) + dy_n(deltaphiPy,idx) + dz_n(deltaphiPz, idx);
		
		mu[idx] = alpha2phi * ddeltaphi;
		mu[idx] -= Kphi * laplacianphi[idx];
		mu[idx] += beta * (4 * ddeltaphi * Pgradphitot - divdeltaphiP);
		//mu[idx] -= 0.5 * alpha2 * Psquare; //removed bulk polarisation
		
		if( phi[idx] > adhesion_threshold ){
			mu[idx] -= delta * ( laplacianphitotal[idxt] - laplacianphi[idx] );
			
		}
		if( phi[idx] > repulsion_threshold ){
			mu[idx] += epsilon * ( phitotal[idxt] - phi[idx] );
		}
	}
	
	
	minidx = l * domain_volume ; 
	maxidx = (l+1) * domain_volume ;
	for(idx = minidx ; idx < maxidx ; idx++ ){
		/*Streaming increment*/
		h_phi[idx] = M * laplacian_n(mu, idx);
		
		vx = w1 * Px[idx];
		vy = w1 * Py[idx];
		vz = w1 * Pz[idx];
		
		vxdphidx = upwind_x(vx, phi, idx);
		vydphidy = upwind_y(vy, phi, idx);
		vzdphidz = upwind_z(vz, phi, idx);
		
		divP = dx_n(Px,idx) + dy_n(Py,idx) + dz_n(Pz,idx); 
		
		h_phi[idx] -= (vxdphidx + vydphidy + vzdphidz);
		h_phi[idx] -= w1 * divP * phi[idx] ;
	}
}

void growth( int l ) {
	int idxt, idx;
	
	for ( idxt = 0 ; idxt < domain_volume ; idxt++ ){
		idx = l * domain_volume + idxt ;
		if( (phi[idx] > growth_threshold_low) && (phitotal[idxt] < growth_threshold_high) ) {
			h_phi[idx] += growthrate * (volume_equil - volume[l] );
		}
	}
}


void P_function( int l ){
	int idx, idxt;
	
	double deltaphi;
	double vx, vy, vz;
	double vxdPxdx, vydPxdy, vzdPxdz;
	double vxdPydx, vydPydy, vzdPydz;
	double vxdPzdx, vydPzdy, vzdPzdz;
	
	for ( idxt = 0 ; idxt < domain_volume ; idxt++ ){
		idx = l*domain_volume + idxt;
		
		deltaphi = phi[idx] * phi[idx] * (phi[idx] - 2.) * (phi[idx] - 2.);
		
		hx[idx] = K * laplacian_n(Px, idx);
		hy[idx] = K * laplacian_n(Py, idx);
		hz[idx] = K * laplacian_n(Pz, idx);
		
		hx[idx] -= alpha4 * Px[idx];
		hy[idx] -= alpha4 * Py[idx];
		hz[idx] -= alpha4 * Pz[idx];
		
		idxt = idx % domain_volume;
		
		hx[idx] -=  beta * deltaphi * dx_n(phitotal,idxt);
		hy[idx] -=  beta * deltaphi * dy_n(phitotal,idxt);
		hz[idx] -=  beta * deltaphi * dz_n(phitotal,idxt);
		
		hx[idx] *= G;
		hy[idx] *= G;
		hz[idx] *= G;
		
		vx = w1 * Px[idx];
		vy = w1 * Py[idx];
		vz = w1 * Pz[idx];
		
		vxdPxdx = upwind_x(vx, Px, idx);
		vydPxdy = upwind_y(vy, Px, idx);
		vzdPxdz = upwind_z(vz, Px, idx);
		
		vxdPydx = upwind_x(vx, Py, idx);
		vydPydy = upwind_y(vy, Py, idx);
		vzdPydz = upwind_z(vz, Py, idx);
		
		vxdPzdx = upwind_x(vx, Pz, idx);
		vydPzdy = upwind_y(vy, Pz, idx);
		vzdPzdz = upwind_z(vz, Pz, idx);
		
		hx[idx] -= (vxdPxdx + vydPxdy + vzdPxdz);
		hy[idx] -= (vxdPydx + vydPydy + vzdPydz);
		hz[idx] -= (vxdPzdx + vydPzdy + vzdPzdz);
	}
	
}

void updatephi0 (int l){
	int idx;
	int minidx, maxidx;
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		phi[idx]  = phiold[idx] + dt * h_phi[idx]; 
	}
	
}

void updateP0 (int l){
	int idx;
	int minidx, maxidx;
	minidx = l * domain_volume;
	maxidx = (l+1) * domain_volume;
	for ( idx = minidx ; idx < maxidx ; idx++ ){
		Px[idx]  = Pxold[idx] + dt * hx[idx];
		Py[idx]  = Pyold[idx] + dt * hy[idx];
		Pz[idx]  = Pzold[idx] + dt * hz[idx];
	}	
}

void division(int l, int n) {
	int idxnew, idxold, idxt;
	int x,y;
	int newl;
	double division_line_x[Lx], division_line_y[Lx];
	
	cell_position(l);
	cell_inertial_axes(l);
	
	phase_counter = phase_counter + 1;
	config_space = phase_counter * Lx * Ly * Lz;
	
	newl = phase_counter - 1;
	
	for( x = 0 ; x < Lx ; x++ ) {
		division_line_x[x] = x;
		division_line_y[x] = Ry[l] +  (x-Rx[l])*((eigen_direction1x[l] > 0 ) ? 1 : -1 ) * (eigen_direction1y[l] / eigen_direction1x[l]);
	}
	
	for( idxt = 0 ; idxt < domain_volume ; idxt++ ) {
		y  = (idxt % LyLz) / Lz;
		x = idxt / LyLz;
		idxold =  l * domain_volume  + idxt;
		idxnew = newl * domain_volume + idxt;
		
		if( y > division_line_y[x] + spacing_division ){
		}else{	
			if( y <= division_line_y[x] - spacing_division ){
				phi[idxnew] = phi[idxold];
				phi[idxold] = 0.;
				
				Px[idxnew] = Px[idxold];
				Px[idxold] = 0.;
				
				Py[idxnew] = Py[idxold];
				Py[idxold] = 0.;
				
				Pz[idxnew] = Pz[idxold];
				Pz[idxold] = 0.;
			}else{
				phi[idxnew] = 0.;
				phi[idxold] = 0.;
				
				Px[idxnew] = 0.;
				Px[idxold] = 0.;
				
				Py[idxnew] = 0.;
				Py[idxold] = 0.;
				
				Pz[idxnew] = 0.;
				Pz[idxold] = 0.;
			}
		}
	}
	
	cell_mass(l);
	cell_mass(newl);
	printf("==============================================================================\nCell %d has successfully divided creating a cell of type %d, at timestep %d\n==============================================================================\n", l , newl, n);
}

void twodimensional(){
	int idx;
	for(idx = 0 ; idx < config_space ; idx++){
		Pz[idx] = 0.;
	}
}

double upwind_x(double vx, double *field, int idx){
	int didxf, didxb; //forward, back
	int didxff, didxbb; //forward-forward, back-back
	
	double vxdfielddx;
	
	didxf = idxf[idx];
	didxb = idxb[idx];
	
	didxff = idxf[didxf];
	didxbb = idxb[didxb];
	
	vxdfielddx = max(vx, 0.0)*(2.0 * field[didxf] + 3.0 * field[idx] - 6.0 * field[didxb] + field[didxbb]) +
		  min(vx, 0.0)*(-field[didxff] + 6.0 * field[didxf] - 3.0 * field[idx] - 2.0 * field[didxb]);
		  
	vxdfielddx *= onesixth;
	
	return(vxdfielddx);
}
double upwind_y(double vy, double *field, int idx){
	int didxr, didxl; //left, right
	int didxrr, didxll; //left-left, right-right
	
	double vydfielddy;
	
	didxr = idxr[idx];
	didxl = idxl[idx];
	
	didxrr = idxr[didxr];
	didxll = idxl[didxl];
	
	vydfielddy = max(vy, 0.0)*(2.0 * field[didxr] + 3.0 * field[idx] - 6.0 * field[didxl] + field[didxll]) +
		  min(vy, 0.0)*(-field[didxrr] + 6.0 * field[didxr] - 3.0 * field[idx] - 2.0 * field[didxl]);
	
	vydfielddy *= onesixth;
	
	return(vydfielddy);
}
double upwind_z(double vz, double *field, int idx){
	int didxu, didxd;//up, down
	int didxuu, didxdd;//up-up, down-down
	
	double vzdfielddz;
	
	didxu = idxu[idx];
	didxd = idxd[idx]; 
	
	didxuu = idxu[didxu];
	didxdd = idxd[didxd]; 
	
	vzdfielddz = max(vz, 0.0)*(2.0 * field[didxu] + 3.0 * field[idx] - 6.0 * field[didxd] + field[didxdd]) +
		  min(vz, 0.0)*(-field[didxuu] + 6.0 * field[didxu] - 3.0 * field[idx] - 2.0 * field[didxd]);
	
	vzdfielddz *= onesixth;
	
	return(vzdfielddz);
}

double dx_n(double *field, int idx){
	int didxf, didxb;
	
	int didxfr, didxfl;
	int didxbr, didxbl;
	
	int didxfu, didxfd;
	int didxbu, didxbd;
	
	double dxfield;
	
	didxf = idxf[idx];
	didxb = idxb[idx];
	
	didxfr = idxr[didxf];
	didxfl = idxl[didxf];
	
	didxbr = idxr[didxb];
	didxbl = idxl[didxb];
	
	didxfu = idxu[didxf];
	didxfd = idxd[didxf];
	
	didxbu = idxu[didxb];
	didxbd = idxd[didxb];
	
	dxfield = (field[didxfu]
		+ field[didxfl] + 2.0 * field[didxf] + field[didxfr]
		+ field[didxfd])
		- (field[didxbu]
		+ field[didxbl] + 2.0 * field[didxb] + field[didxbr]
		+ field[didxbd]);
		
	dxfield *= onetwelth;
	
	return(dxfield);
}
double dy_n(double *field, int idx){
	int didxr, didxl;
	
	int didxrf, didxrb;
	int didxlf, didxlb;
	
	int didxru, didxrd;
	int didxlu, didxld;
	
	double dyfield;
	
	didxr = idxr[idx];
	didxl = idxl[idx];
	
	didxrf = idxf[didxr];
	didxrb = idxb[didxr];
	
	didxlf = idxf[didxl];
	didxlb = idxb[didxl];
	
	didxlu = idxu[didxl];
	didxld = idxd[didxl];
	
	didxru = idxu[didxr];
	didxrd = idxd[didxr];
	
	dyfield = (field[didxru]
		+ field[didxrb] + 2.0 * field[didxr] + field[didxrf]
		+ field[didxrd])
		- (field[didxlu]
		+ field[didxlb] + 2.0 * field[didxl] + field[didxlf]
		+ field[didxld]) ; 
		
	dyfield *= onetwelth;
	
	return(dyfield);
}
double dz_n(double *field, int idx){
	int didxu, didxd;
	
	int didxuf, didxub;
	int didxdf, didxdb;
	
	int didxur, didxul;
	int didxdr, didxdl;
	
	double dzfield;
	
	didxu = idxu[idx];
	didxd = idxd[idx];
	
	didxuf = idxf[didxu];
	didxub = idxb[didxu];
	
	didxdf = idxf[didxd];
	didxdb = idxb[didxd];
	
	didxur = idxr[didxu];
	didxul = idxl[didxu];
	
	didxdr = idxr[didxd];
	didxdl = idxl[didxd];
	
	dzfield = (field[didxur]
		+ field[didxub] + 2.0 * field[didxu] + field[didxuf]
		+ field[didxul]) 
		- (field[didxdr]
		+ field[didxdb] + 2.0 * field[didxd] + field[didxdf]
		+ field[didxdr]) ;
	
	dzfield *= onetwelth;
	
	return(dzfield);	
}

double laplacian_n(double *field, int idx){
	int didxf, didxb;
	int didxr, didxl;
	int didxu, didxd;
	
	int didxfr, didxfl, didxfu, didxfd;
	int didxbr, didxbl, didxbu, didxbd;
	
	int didxru, didxrd;
	int didxlu, didxld;
	
	double laplacian;
	
	didxf = idxf[idx];
	didxb = idxb[idx];
	
	didxfr = idxr[didxf];
	didxbr = idxr[didxb];
	
	didxfl = idxl[didxf];
	didxbl = idxl[didxb];
	
	didxfu = idxu[didxf];
	didxbu = idxu[didxb];

	didxfd = idxd[didxf];
	didxbd = idxd[didxb];
	
	didxr = idxr[idx];
	didxl = idxl[idx];
	
	didxru = idxu[didxr];
	didxlu = idxu[didxl];
	
	didxrd = idxd[didxr];
	didxld = idxd[didxl];
	
	didxu = idxu[idx];
	didxd = idxd[idx];
	
	//printf("%d %d %d %d %d %d %d %d %d %d %d\n",idx, didxf, didxb, didxr, didxl, didxu, didxd, didxfr, didxfl, didxbr, didxbl);
	/*
	if(idx/domain_volume>0){
		//printf("PDO\n");
	}
	laplacian = phi[didxfu];
	if(idx/domain_volume>0){
		//printf("PDO  %d %d %.3E\n", phase_counter, idx, laplacian);
	}
	laplacian +=  phi[didxfl];
	laplacian +=  2.0 * phi[didxf];
	laplacian +=  phi[didxfr];
	laplacian +=  phi[didxfd];
	
	laplacian += phi[didxlu] ;
	laplacian += 2.0 * phi[didxu];
	laplacian += phi[didxru];
	laplacian += 2.0 * phi[didxl];
	laplacian += - 24.0 * phi[idx];
	laplacian += 2.0 * phi[didxr];
	laplacian += phi[didxld];
	laplacian += 2.0 * phi[didxd];
	laplacian += phi[didxrd];
	laplacian += phi[didxbu];
	laplacian += phi[didxbl];
	laplacian +=  2.0 * phi[didxb];
	laplacian += phi[didxbr];
	laplacian += phi[didxbd];	
	*/
	
	laplacian =   (field[didxfu]
			+ field[didxfl] + 2.0 * field[didxf] + field[didxfr]
			+ field[didxfd]) 

			+ (field[didxlu] + 2.0 * field[didxu] + field[didxru]
			+ 2.0 * field[didxl] - 24.0 * field[idx] + 2.0 * field[didxr]
			+ field[didxld] + 2.0 * field[didxd] + field[didxrd]) 
			
			+ (field[didxbu]
			+ field[didxbl] + 2.0 * field[didxb] + field[didxbr]
			+ field[didxbd]) ;
	
	laplacian = laplacian * onesixth;
	/*
	if(idx/domain_volume>0){
		printf("Sono arrivato qui %d %d %.3E\n", phase_counter, idx, laplacian);
	}*/
	 
	return(laplacian);
}


double max(double x, double y){
	if (x > y) {
		return x;
	} else {
		return y;
	}
}


double min(double x, double y){
	if (x > y) {
		return y;
	} else {
		return x;
	}
}

void plots(int n){
	#if PLOT_FIELDS
		plot_fields(n);
	#endif
	print_to_screen(n);
	//plot_laplacians(n);
	#if PLOT_POSITION
		plot_position(n);
	#endif
	#if PLOT_VELOCITY
		plot_velocity(n);
	#endif
	plot_wound_size(n);
}


void plot_fields(int n){
	int x,y,z, l;
	int idx, idxt;

	double Pxtotal, Pytotal, Pztotal;
	double densityinterpol;
	double Pmod;
	char filename[50];
	FILE *output;

	sprintf(filename, "plot.txt.%d", n);

	output = fopen(filename, "w");
	
	for (x = 0; x < Lx; x++) {
		for (y = 0; y < Ly; y++) {
			for (z = 0; z < Lz; z++) {
				Pxtotal = 0.0;
				Pytotal = 0.0;
				Pztotal = 0.0;
				
				idxt = LyLz * x + Lz * y + z; 
				
				for (l = 0; l < phase_counter; l++) {
					idx = domain_volume * l + idxt; 
					Pxtotal += Px[idx];
					Pytotal += Py[idx];
					Pztotal += Pz[idx];
				}
				densityinterpol = 0.25 * phitotal[idxt] * phitotal[idxt] * (3. - phitotal[idxt]);
				Pmod = Pxtotal*Pxtotal + Pytotal*Pytotal + Pztotal*Pztotal;
				fprintf(output, "%d %d %d %E %E %E %E %E %E",
						  x, y, z,
						  Pxtotal, Pytotal, Pztotal,
						  phitotal[idxt], densityinterpol, Pmod);
				//fprintf(output,"\n");
			}
			fprintf(output, "\n");
		}
		fprintf(output, "\n");
	}
	fclose(output); 
}

void plot_wound_size(int n){
	FILE *output;
	char woundfile[50];
	
	sprintf(woundfile, "woundsize.dat.%E", w1);
	if( n == 0 ){
		output = fopen(woundfile, "w");
	}else{
		output = fopen(woundfile, "a");
	}
	fprintf(output, " %d %d\n", n, empty_counter );
	
	fclose(output);	
}

void plot_laplacians(int n){
	int x,y,z, l;
	int idx, idxt;
	char filename[50];
	FILE *output;

	sprintf(filename, "lap_plot.txt.%d", n);

	output = fopen(filename, "w");
	
	for (x = 0; x < Lx; x++) {
		for (y = 0; y < Ly; y++) {
			for (z = 0; z < Lz; z++) {
				idxt = LyLz * x + Lz * y + z; 
				fprintf(output, "%d %d %d %E",
						  x, y, z,laplacianphitotal[idxt]);
				for(l = 0 ; l < phase_counter ; l++ ){
					idx = l * domain_volume + idxt;
					fprintf(output, " %E", mu[idx]);
				}
				//fprintf(output,"\n");
			}
			fprintf(output, "\n");
		}
		fprintf(output, "\n");
	}
	fclose(output); 
}

void print_to_screen(int n) {
	int l, print_volumes = 1;
	printf("%d %d", n, phase_counter);
	
	if(print_volumes == 1){
		for( l = 0 ; l < phase_counter ; l++ ) {
			printf("\t%.3E", volume[l]);
		}
	}
	printf("\n");
}

void plot_position(int n) {
	int l;
	FILE *output;
	if(n==0){
		output = fopen("positions.dat", "w");
	}else{
		output = fopen("positions.dat", "a");
	}
	fprintf(output, " %d", n);
	for( l = 0 ; l < phase_counter ; l++ ) {
		fprintf(output, "\t%.3E\t%.3E\t%.3E", Rx[l], Ry[l], Rz[l]);
	}
	fprintf(output, "\n");
	fclose(output);
}

void plot_velocity(int n) {
	int l;
	FILE *output;
	
	output = fopen("velocity.dat", "a");
	fprintf(output, " %d", n);
	for( l = 0 ; l < phase_counter ; l++ ) {
		fprintf(output, "\t%.3E\t%.3E\t%.3E", Vx[l], Vy[l], Vz[l]);
	}
	fprintf(output, "\n");
	fclose(output);
}

void plot_initial_config(void){
	int x,y,z, l;
	int idx, idxt;

	double Pxtotal, Pytotal, Pztotal;
	double densityinterpol;
	double Pmod;
	char filename[50];
	FILE *output;

	sprintf(filename, "initial.plot.%e", w1);

	output = fopen(filename, "w");
	
	for (x = 0; x < Lx; x++) {
		for (y = 0; y < Ly; y++) {
			for (z = 0; z < Lz; z++) {
				Pxtotal = 0.0;
				Pytotal = 0.0;
				Pztotal = 0.0;
				
				idxt = LyLz * x + Lz * y + z; 
				
				for (l = 0; l < phase_counter; l++) {
					idx = domain_volume * l + idxt; 
					Pxtotal += Px[idx];
					Pytotal += Py[idx];
					Pztotal += Pz[idx];
				}
				densityinterpol = 0.25 * phitotal[idxt] * phitotal[idxt] * (3. - phitotal[idxt]);
				Pmod = Pxtotal*Pxtotal + Pytotal*Pytotal + Pztotal*Pztotal;
				fprintf(output, "%d %d %d %E %E %E %E %E %E",
						  x, y, z,
						  Pxtotal, Pytotal, Pztotal,
						  phitotal[idxt], densityinterpol, Pmod);
				//fprintf(output,"\n");
			}
			fprintf(output, "\n");
		}
		fprintf(output, "\n");
	}
	fclose(output); 
}

void plot_final_config(int n){
	int x,y,z, l;
	int idx, idxt;

	double Pxtotal, Pytotal, Pztotal;
	double densityinterpol;
	double Pmod;
	char filename[50];
	FILE *output;

	sprintf(filename, "final.plot.%.1e.%d", w1, n);

	output = fopen(filename, "w");
	
	for (x = 0; x < Lx; x++) {
		for (y = 0; y < Ly; y++) {
			for (z = 0; z < Lz; z++) {
				Pxtotal = 0.0;
				Pytotal = 0.0;
				Pztotal = 0.0;
				
				idxt = LyLz * x + Lz * y + z; 
				
				for (l = 0; l < phase_counter; l++) {
					idx = domain_volume * l + idxt; 
					Pxtotal += Px[idx];
					Pytotal += Py[idx];
					Pztotal += Pz[idx];
				}
				densityinterpol = 0.25 * phitotal[idxt] * phitotal[idxt] * (3. - phitotal[idxt]);
				Pmod = Pxtotal*Pxtotal + Pytotal*Pytotal + Pztotal*Pztotal;
				fprintf(output, "%d %d %d %E %E %E %E %E %E",
						  x, y, z,
						  Pxtotal, Pytotal, Pztotal,
						  phitotal[idxt], densityinterpol, Pmod);
				//fprintf(output,"\n");
			}
			fprintf(output, "\n");
		}
		fprintf(output, "\n");
	}
	fclose(output); 
}

void checkpoint(int n) {
	int x,y,z, l;
	int idx, idxt;
	double Pxl, Pyl, Pzl;
	double phil;
	char filename1[50];
	FILE *output1;
	
	sprintf(filename1, "checkpoint.txt.%d", n);

	output1 = fopen(filename1, "w");

	for (x = 0; x < Lx; x++) {
		for (y = 0; y < Ly; y++) {
			for (z = 0; z < Lz; z++) {
				fprintf(output1, " %d\t%d\t%d", x, y, z);
				for (l = 0; l < phase_counter; l++) {
					idx = l * domain_volume + x * LyLz + y * Lz + z ;
					Pxl = Px[idx];
					Pyl = Py[idx];
					Pzl = Pz[idx];
					phil = phi[idx];
					fprintf(output1, "\t%E\t%E\t%E\t%E", Pxl, Pyl, Pzl, phil);
				}
				//fprintf(output1, "\n");
			}
			fprintf(output1, "\n");
		}
		fprintf(output1, "\n");
	}
	fclose(output1);
	printf("===================================================\nConfiguration successfully saved @iteration = %d\n===================================================\n", n);
}
