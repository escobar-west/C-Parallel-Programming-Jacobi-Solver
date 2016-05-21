
//JACOBI PARALLEL CODE

#include <iostream>
#include <fstream>
#include <cmath>
#include "mpi.h"
using namespace std;

int main( int argc, char * argv[] ) {

int N    = 256; // number of x, y coordinates on mesh
double T = 1.0;
double err_tot = 1.0e-7;
int max_iter = 1.0e5;
double h = 2.0*T/(N-1);
double pi = 3.141592653;

int rank, size;
MPI_Status status;

MPI_Init( &argc, &argv );

MPI_Comm_rank( MPI_COMM_WORLD, &rank );
MPI_Comm_size( MPI_COMM_WORLD, &size );

if( N % size != 0 )
	MPI_Abort( MPI_COMM_WORLD, 1 );

MPI_Barrier(MPI_COMM_WORLD);

int numRows = N/size; //number of rows that each process is responsible for;

double x_loc[numRows + 2][N]; //two phantom nodes, real nodes are indexed 1,...,numRows
double x_new[numRows][N];

for( int i=0; i<numRows + 2; i++ ) {
	for( int j=0; j<N; j++ ) {
		x_loc[i][j] = 0;
	}
}

int i_first = (rank == 0) ? 2 : 1;
int i_last  = (rank == size-1) ? numRows-1 : numRows;

double diff_loc = 0;
double diff_globe = 0;
int count = 0;

double start = MPI_Wtime();
do {
	if( rank < size-1 ) { //send up
		MPI_Send( x_loc[numRows], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD );
	}
	if( rank > 0 ) { //recieve from down
		MPI_Recv( x_loc[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status );
	}
	if( rank > 0 ) { // send down
		MPI_Send( x_loc[1], N, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD );
	}
	if( rank < size-1 ) { //receive from up
		MPI_Recv( x_loc[numRows+1], N, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status );
	}
	
	diff_loc = 0;

	for( int i=i_first; i<=i_last; i++ ) {
		for( int j=1; j<N-1; j++ ) {

			x_new[i-1][j] =  ( x_loc[i][j+1] 
						+ x_loc[i][j-1]
					 	+ x_loc[i-1][j] 
						+ x_loc[i+1][j]
						- h*h*x_loc[i][j]
					 	- h*h*sin(pi*(h*(rank*numRows+i-1)-T))*sin(pi*(h*j-T)) )/4.0;
	
			diff_loc += (x_new[i-1][j] - x_loc[i][j])*(x_new[i-1][j] - x_loc[i][j]);
		}	
	}	
	for( int i=i_first; i<=i_last; i++ ) {
		for( int j=1; j<N-1; j++ ) {
			x_loc[i][j] = x_new[i-1][j];
		}
	}
	MPI_Allreduce( &diff_loc, &diff_globe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
	diff_globe = sqrt( diff_globe );
	count++;
} while( diff_globe > err_tot && count < max_iter );

double end = MPI_Wtime();
double x_final[N][N];

MPI_Gather( x_loc[1], N*numRows, MPI_DOUBLE, x_final, N*numRows, MPI_DOUBLE, 0, MPI_COMM_WORLD );
if( rank == 0 ) {
	ofstream outFile;
	outFile.open("data.txt");

	outFile << N << " " << T << " " << h << " ";
	
	for( int i=0; i<N; i++ ) {
		for( int j=0; j<N; j++ ) {
			outFile << x_final[i][j] << " ";
		}
	}
}

MPI_Barrier(MPI_COMM_WORLD);

if( rank == 0) cout << "Total time is: " << end - start << endl;

MPI_Finalize();
return 0;
}
