/*
 * Feb. 3, 2017 David Z
 *
 * Check whether the covariance Matrix is rank deficiency 
 *
 * */

#ifndef COVARIANCE_CHECK_H
#define COVARIANCE_CHECK_H 


#include <cmath>

namespace VRO
{
template<typename M>
bool MatrixEqual(M& m1, M& m2, double tolerance)
{
  if(m1.rows() != m2.rows() || m1.cols() != m2.cols()) return false; 
  for(int i=0; i<m1.rows(); i++)
    for(int j=0; j<m1.cols(); j++)
    {
      if(fabs(m1(i,j) - m2(i,j)) > tolerance) 
        return false; 
    }
  return true; 
}

template<typename M>
void reducePrecision(M& m, double precision, double precision_inv)
{
  for(int i=0; i<m.rows(); i++)
    for(int j=0; j<m.cols(); j++)
    {
      m(i, j) = (double)((int)(m(i,j)*precision))*precision_inv;
    }
  return;
}

}

#endif
