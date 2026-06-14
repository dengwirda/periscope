
C     Interface to 1d and 2d "Akima" cubic interpolation

C     1D scheme
      SUBROUTINE interp1d_(ORD,NDP,XD,YD,NIP,XI,YI,IER)
C     ORD  order of interpolant
C     NDP  no. points to interpolate from
C     XD   x-value of grid
C     YD   value of function on grid
C     NIP  no. points to interpolate onto
C     XI   x-value of output
C     YI   value of function at output
C     IER  return code (IER==0 on success)
      INTEGER, INTENT(IN) :: ORD,NDP,NIP
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: XD, YD
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: XI
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: YI
      INTEGER, INTENT(INOUT) :: IER 
      CALL UVIP3P(ORD,NDP,XD,YD,NIP,XI,YI,IER)
      END SUBROUTINE

C     2D scheme
      SUBROUTINE interp2d_(NXD,NYD,XD,YD,ZD,NIP,XI,YI,ZI,IER,WK)
C     NXD  no. x-grid to interpolate from
C     NYD  no. y-grid to interpolate from
C     XD   x-value of grid
C     YD   y-value of grid
C     ZD   value of function on grid
C     NIP  no. points to interpolate onto
C     XI   x-value of output
C     YI   y-value of output
C     ZI   value of function at output
C     IER  return code (IER==0 on success)
C     WK   work array, length NXD*NYD*3
      INTEGER, INTENT(IN) :: NXD,NYD,NIP
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: XD
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: YD
      DOUBLE PRECISION, DIMENSION(:,:), INTENT(IN) :: ZD
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: XI, YI
      DOUBLE PRECISION, DIMENSION(:), INTENT(INOUT) :: ZI
      INTEGER, INTENT(INOUT) :: IER
      DOUBLE PRECISION, DIMENSION(:), INTENT(IN) :: WK
      CALL RGBI3P(1,NXD,NYD,XD,YD,ZD,NIP,XI,YI,ZI,IER,WK)
      END SUBROUTINE

