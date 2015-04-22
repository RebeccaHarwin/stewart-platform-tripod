#Copyright (c) <year>, <copyright holder>
#All rights reserved.

#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#    * Neither the name of the <organization> nor the
#      documentation and/or other materials provided with the distribution.
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


try:
    import numpy as np
except:
    import scisoftpy as np
pi = np.pi


def norm(a):
    return np.sqrt(np.dot(a, a))


def cross(a, b):
    c = np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])
    return c


class tripod_class():
    '''
    Python class to carry out calculations for tripod with 6 base translations
    See tripod_doc.texw for description - a pweave document.
    Needs NumPy
    Initialization arguments:
    l, t, psi:      Lists of l, t, psi values for each leg
    c:              x,y,z coordinates of tooling point relative to the top
    theta:          Approximate theta values (used for signs)
    BX,BY:         Lists X and Y coordinates for base positions [0,0,0],[0,0,0]
    Main user (public) methods:
    Typing object name (string representation) gives a summary of parameters
        and calculation of tooling point parameters from base vectors and back
    self.ctool((X1, X2, X3), (Y1, Y2, Y3) calculates tool parameters from
        X list (X values for each leg) and Y list;
        outputs coordinates of tooling point and tilt angles (degrees)
    self.cbase((CX, CY, CZ),(alpha1,alpha2,alpha3)) calculates base vectors for
        tooling point coordinates and angles (degrees)
    '''

    def __init__(self, l=[134.2, 134.2, 134.2], t=[219.129, 219.129, 84.963], psi=[-pi/3, pi/3, 0], c=[150.102, 84.9634/2, 35.7574], theta=[pi/4, pi/4, -pi/4], BX=[0.0, 0.0, 357.31303], BY=[249.32458, 0.0, 249.32458/2], P=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], G=[0.0, 0.0, 0.0, 0.0, 0.0], H=[0.0, 0.0, 0.0, 0.0, 0.0], I=[0.0, 0.0, 0.0, 0.0, 0.0]):
        # l3 = 134.200007617 may reduce errors, as might t1=219.129002753.
        # Values of parameters when tripod is in 'zero position'.
        self.l, self.t, self.psi, self.c, self.theta, self.BX, self.BY, self.P, self.G, self.H, self.I = l, t, psi, c, theta, BX, BY, P, G, H, I
        self.X, self.Y = 0, 0  # Initial values of slides.

    def __repr__(self):
        rad_to_deg = 180.0/np.pi
        C, (alpha1, alpha2, alpha3) = self.ctool(self.X, self.Y)
        # Calculate top position and angles from base translations.
        X, Y = self._calcBXY(C, np.array([alpha1, alpha2, alpha3])*np.pi/180)
        # Reverse calculation (as a check).

        return '\nTripod parameters:\n\n' \
            + 'Leg lengths(l):\t\t %.5f, %.5f, %.5f\n' % tuple(self.l) \
            + 'Edge lengths(s):\t\t %.5f, %.5f, %.5f\n' % tuple(self.t) \
            + 'Hinge rotations (psi deg):\t %.5f, %.5f, %.5f\n' % tuple(np.array(self.psi)*rad_to_deg) \
            + 'Tool point xyz:\t\t %.5f, %.5f, %.5f\n' % tuple(self.c) \
            + 'Base X centres:\t\t %.5f, %.5f, %.5f\n' % tuple(self.BX) \
            + 'Base Y centres:\t\t %.5f, %.5f, %.5f\n' % tuple(self.BY) \
            + 'Approx. tilts (theta deg ):\t %.5f, %.5f, %.5f\n\n' % tuple(np.array(self.theta)*rad_to_deg) \
            + 'Slide X:\t\t\t %.5f, %.5f, %.5f\n' % tuple(self.X) \
            + 'Slide Y:\t\t\t %.5f, %.5f, %.5f\n' % tuple(self.Y) \
            + 'Tool point (C):\t\t %.5f, %.5f, %.5f\n' % tuple(C) \
            + 'Tool angles (alpha deg):\t %.5f, %.5f, %.5f\n' % (alpha1, alpha2, alpha3) \
            + 'Reverse calc Slide X:\t\t %.5f, %.5f, %.5f\n' % tuple(X) \
            + 'Reverse calc Slide Y:\t\t %.5f, %.5f, %.5f\n' % tuple(Y) \


    def ctool(self, X, Y):
        '''self.ctool((X1, X2, X3), (Y1, Y2, Y3) calculates tool parameters from
            X list (X values for each leg) and Y list
            outputs coordinates of tooling point and tilt angles (degrees)'''
        self.X, self.Y = X, Y
        # Main calculation of C and alpha tilts from base positions.
        # Base vectors from given translations (X,Y) and fixed centres (BX, BY).
        B1 = np.array((X[0]+self.BX[0], Y[0]+self.BY[0], 0))
        B2 = np.array((X[1]+self.BX[1], Y[1]+self.BY[1], 0))
        B3 = np.array((X[2]+self.BX[2], Y[2]+self.BY[2], 0))
        # Top (T) vectors from bottom (B) vectors and leg vectors (v).
        psi1 = self.psi[0]
        psi2 = self.psi[1]
        psi3 = self.psi[2]
        l1 = self.l[0]
        l2 = self.l[1]
        l3 = self.l[2]
        t1 = self.t[0]
        t2 = self.t[1]
        t3 = self.t[2]
        (theta1, theta2, theta3) = self._calcThetafromB([B1, B2, B3], psi1, psi2, psi3, l1, l2, l3, t1, t2, t3, X, Y)
        v1 = np.array([np.cos(self.psi[0])*np.sin(theta1), np.sin(self.psi[0])*np.sin(theta1), np.cos(theta1)])
        v2 = np.array([np.cos(self.psi[1])*np.sin(theta2), np.sin(self.psi[1])*np.sin(theta2), np.cos(theta2)])
        v3 = np.array([np.cos(self.psi[2])*np.sin(theta3), np.sin(self.psi[2])*np.sin(theta3), np.cos(theta3)])
        T1, T2, T3 = B1+self.l[0]*v1, B2+self.l[1]*v2, B3+self.l[2]*v3
        self.T1, self.T2, self.T3 = T1, T2, T3
        # Calculate tooling point coordinates and angles.
        yvec = (T1-T2)/norm(T1-T2)
        zvec = cross((T3-T1), yvec)/norm(cross((T3-T1), yvec))
        xvec = cross(yvec, zvec)
        C = T2 + self.c[0]*xvec + self.c[1]*yvec + self.c[2]*zvec
        # Calculate alpha angles.
        self.alpha1 = np.arctan(yvec[2]/zvec[2])
        self.alpha2 = np.arcsin(-xvec[2])
        self.alpha3 = np.arctan(xvec[1]/xvec[0])
        return (C, np.array([self.alpha1*180/np.pi, self.alpha2*180/np.pi, self.alpha3*180/np.pi]))

    def _calcThetafromB(self, B, psi1, psi2, psi3, l1, l2, l3, t1, t2, t3, X, Y):
        # Calculates leg tilt angles (see documentation).
        # Uses Numpy's roots solver, a variant of which has been created in dnp.
        # The solver solves the 8th order polynomial in x**2 that results.
        B1 = B[0]
        B2 = B[1]
        B3 = B[2]
        B1X, B1Y = B1[0], B1[1]
        B2X, B2Y = B2[0], B2[1]
        B3X, B3Y = B3[0], B3[1]

        def calc_parameters(B1X, B1Y, B2X, B2Y, B3X, B3Y, psi1, psi2, psi3, l1, l2, l3, t1, t2, t3):
            # All params: theta1, theta2, theta3, B1X, B1Y, B2X, B2Y, B3X, B3Y, psi1, psi2, psi3, l1, l2, 3, t1, t2, t3.
            # Calculates parameters as described in report and documentation.
            D1 = (-2*l1*((B2X-B1X)*np.cos(psi1) + (B2Y-B1Y)*np.sin(psi1)))
            D2 = (2*l2*((B2X-B1X)*np.cos(psi2) + (B2Y-B1Y)*np.sin(psi2)))
            D3 = -2*l1*l2*(np.cos(psi1)*np.cos(psi2)+np.sin(psi1)*np.sin(psi2))
            D4 = (-2*l1*l2)
            D5 = ((B2X-B1X)**2 + (B2Y-B1Y)**2 + l1**2 + l2**2 - t3**2)
            E1 = (-2*l2*((B3X-B2X)*np.cos(psi2)+(B3Y-B2Y)*np.sin(psi2)))
            E2 = (2*l3*((B3X-B2X)*np.cos(psi3) + (B3Y-B2Y)*np.sin(psi3)))
            E3 = -2*l2*l3*(np.cos(psi2)*np.cos(psi3)+np.sin(psi2)*np.sin(psi3))
            E4 = -2*l2*l3
            E5 = ((B3X-B2X)**2 + (B3Y-B2Y)**2 + l2**2+l3**2-t1**2)
            F1 = (-2*l3*((B1X-B3X)*np.cos(psi3) + (B1Y-B3Y)*np.sin(psi3)))
            F2 = (2*l1*((B1X-B3X)*np.cos(psi1)+(B1Y-B3Y)*np.sin(psi1)))
            F3 = -2*l1*l3*(np.cos(psi1)*np.cos(psi3)+np.sin(psi1)*np.sin(psi3))
            F4 = (-2*l1*l3)
            F5 = ((B1X-B3X)**2+(B1Y-B3Y)**2 + l1**2 + l3**2 - t2**2)
            G1 = -D1-D2+D3+D5
            G2 = D1-D2-D3+D5
            G3 = 4*D4
            G4 = -D1+D2-D3+D5
            G5 = D1+D2+D3+D5
            H1 = -E1-E2+E3+E5
            H2 = E1-E2-E3+E5
            H3 = 4*E4
            H4 = -E1+E2-E3+E5
            H5 = E1+E2+E3+E5
            I1 = -F1-F2+F3+F5
            I2 = F1-F2-F3+F5
            I3 = 4*F4
            I4 = -F1+F2-F3+F5
            I5 = F1+F2+F3+F5
            K1 = (G1**2)*(H2**2)-2*G1*G4*H1*H2+(G4**2)*(H1**2)
            K2 = 2*G1*G2*(H2**2)-2*G1*G5*H1*H2-2*G2*G4*H1*H2+2*G4*G5*(H1**2)+(G3**2)*H1*H2
            K3 = (G2**2)*(H2**2)-2*G2*G5*H1*H2+(G5**2)*(H1**2)
            K4 = -G1*G3*H2*H3-G3*G4*H1*H3
            K5 = -G2*G3*H2*H3-G3*G5*H1*H3
            K6 = 2*(G1**2)*H2*H5-2*G1*G4*H2*H4-2*G1*G4*H1*H5+2*(G4**2)*H1*H4+G1*G4*(H3**2)
            K7 = 4*G1*G2*H2*H5-2*G1*G5*H2*H4-2*G1*G5*H1*H5-2*G2*G4*H2*H4-2*G2*G4*H1*H5+4*G4*G5*H1*H4+(G3**2)*H2*H4 +(G3**2)*H1*H5+G2*G4*(H3**2)+G1*G5*(H3**2)
            K8 = 2*(G2**2)*H2*H5-2*G2*G5*H2*H4-2*G2*G5*H1*H5+2*H1*H4*(G5**2)+G2*G5*(H3**2)
            K9 = -G1*G3*H3*H5-G3*G4*H3*H4
            K10 = -G2*G3*H3*H5-G3*G5*H3*H4
            K11 = (G1**2)*(H5**2)-2*G1*G4*H4*H5+(G4**2)*(H4**2)
            K12 = 2*G1*G2*(H5**2)-2*G1*G5*H4*H5-2*G2*G4*H4*H5+2*G4*G5*(H4**2)+(G3**2)*H4*H5
            K13 = (G2**2)*(H5**2)-2*G2*G5*H4*H5+(G5**2)*(H4**2)
            P16 = (-I2**4*K1**2-2*I1**2*I2**2*K1*K11-I1**4*K11**2+2*I1*I2**3*K1*K6+2*I1**3*I2*K11*K6-I1**2*I2**2*K6**2)
            P14 = (-4*I2**3*I5*K1**2+4*I1*I2*I3**2*K1*K11-4*I1*I2**2*I4*K1*K11-4*I1**2*I2*I5*K1*K11-4*I1**3*I4*K11**2-2*I1**2*I2**2*K1*K12-2*I1**4*K11*K12-2*I2**4*K1*K2-2*I1**2*I2**2*K11*K2+I2**3*I3*K1*K4-3*I1**2*I2*I3*K11*K4-I1*I2**3*K4**2-I2**2*I3**2*K1*K6+2*I2**3*I4*K1*K6+6*I1*I2**2*I5*K1*K6-I1**2*I3**2*K11*K6+6*I1**2*I2*I4*K11*K6+2*I1**3*I5*K11*K6+2*I1**3*I2*K12*K6+2*I1*I2**3*K2*K6+I1*I2**2*I3*K4*K6-2*I1*I2**2*I4*K6**2-2*I1**2*I2*I5*K6**2+2*I1*I2**3*K1*K7+2*I1**3*I2*K11*K7-2*I1**2*I2**2*K6*K7-3*I1*I2**2*I3*K1*K9+I1**3*I3*K11*K9+2*I1**2*I2**2*K4*K9+I1**2*I2*I3*K6*K9-I1**3*I2*K9**2)
            P12 = (-6*I2**2*I5**2*K1**2-3*I1*I2**2*I3*K1*K10-I3**4*K1*K11+4*I2*I3**2*I4*K1*K11-2*I2**2*I4**2*K1*K11+4*I1*I3**2*I5*K1*K11-8*I1*I2*I4*I5*K1*K11-2*I1**2*I5**2*K1*K11+I1**3*I3*K10*K11-6*I1**2*I4**2*K11**2+4*I1*I2*I3**2*K1*K12-4*I1*I2**2*I4*K1*K12-4*I1**2*I2*I5*K1*K12-8*I1**3*I4*K11*K12-I1**4*K12**2-2*I1**2*I2**2*K1*K13-2*I1**4*K11*K13-8*I2**3*I5*K1*K2+4*I1*I2*I3**2*K11*K2-4*I1*I2**2*I4*K11*K2-4*I1**2*I2*I5*K11*K2-2*I1**2*I2**2*K12*K2-I2**4*K2**2-2*I2**4*K1*K3-2*I1**2*I2**2*K11*K3+3*I2**2*I3*I5*K1*K4+2*I1**2*I2**2*K10*K4+I1*I3**3*K11*K4-6*I1*I2*I3*I4*K11*K4-3*I1**2*I3*I5*K11*K4-3*I1**2*I2*I3*K12*K4+I2**3*I3*K2*K4-I2**3*I4*K4**2-3*I1*I2**2*I5*K4**2+I2**3*I3*K1*K5-3*I1**2*I2*I3*K11*K5-2*I1*I2**3*K4*K5-2*I2*I3**2*I5*K1*K6+6*I2**2*I4*I5*K1*K6+6*I1*I2*I5**2*K1*K6+I1**2*I2*I3*K10*K6-2*I1*I3**2*I4*K11*K6+6*I1*I2*I4**2*K11*K6+6*I1**2*I4*I5*K11*K6-I1**2*I3**2*K12*K6+6*I1**2*I2*I4*K12*K6+2*I1**3*I5*K12*K6+2*I1**3*I2*K13*K6-I2**2*I3**2*K2*K6+2*I2**3*I4*K2*K6+6*I1*I2**2*I5*K2*K6+2*I1*I2**3*K3*K6+I2**2*I3*I4*K4*K6+2*I1*I2*I3*I5*K4*K6+I1*I2**2*I3*K5*K6-I2**2*I4**2*K6**2-4*I1*I2*I4*I5*K6**2-I1**2*I5**2*K6**2-I2**2*I3**2*K1*K7+2*I2**3*I4*K1*K7+6*I1*I2**2*I5*K1*K7-I1**2*I3**2*K11*K7+6*I1**2*I2*I4*K11*K7+2*I1**3*I5*K11*K7+2*I1**3*I2*K12*K7+2*I1*I2**3*K2*K7+I1*I2**2*I3*K4*K7-4*I1*I2**2*I4*K6*K7-4*I1**2*I2*I5*K6*K7-I1**2*I2**2*K7**2+2*I1*I2**3*K1*K8+2*I1**3*I2*K11*K8-2*I1**2*I2**2*K6*K8+I2*I3**3*K1*K9-3*I2**2*I3*I4*K1*K9-6*I1*I2*I3*I5*K1*K9-2*I1**3*I2*K10*K9+3*I1**2*I3*I4*K11*K9+I1**3*I3*K12*K9-3*I1*I2**2*I3*K2*K9-I1*I2*I3**2*K4*K9+4*I1*I2**2*I4*K4*K9+4*I1**2*I2*I5*K4*K9+2*I1**2*I2**2*K5*K9+2*I1*I2*I3*I4*K6*K9+I1**2*I3*I5*K6*K9+I1**2*I2*I3*K7*K9-3*I1**2*I2*I4*K9**2-I1**3*I5*K9**2)
            P10 = (-4*I2*I5**3*K1**2+I2*I3**3*K1*K10-3*I2**2*I3*I4*K1*K10-6*I1*I2*I3*I5*K1*K10-I1**3*I2*K10**2+4*I3**2*I4*I5*K1*K11-4*I2*I4**2*I5*K1*K11-4*I1*I4*I5**2*K1*K11+3*I1**2*I3*I4*K10*K11-4*I1*I4**3*K11**2-I3**4*K1*K12+4*I2*I3**2*I4*K1*K12-2*I2**2*I4**2*K1*K12+4*I1*I3**2*I5*K1*K12-8*I1*I2*I4*I5*K1*K12-2*I1**2*I5**2*K1*K12+I1**3*I3*K10*K12-12*I1**2*I4**2*K11*K12-4*I1**3*I4*K12**2+4*I1*I2*I3**2*K1*K13-4*I1*I2**2*I4*K1*K13-4*I1**2*I2*I5*K1*K13-8*I1**3*I4*K11*K13-2*I1**4*K12*K13-12*I2**2*I5**2*K1*K2-3*I1*I2**2*I3*K10*K2-I3**4*K11*K2+4*I2*I3**2*I4*K11*K2-2*I2**2*I4**2*K11*K2+4*I1*I3**2*I5*K11*K2-8*I1*I2*I4*I5*K11*K2-2*I1**2*I5**2*K11*K2+4*I1*I2*I3**2*K12*K2-4*I1*I2**2*I4*K12*K2-4*I1**2*I2*I5*K12*K2-2*I1**2*I2**2*K13*K2-4*I2**3*I5*K2**2-8*I2**3*I5*K1*K3+4*I1*I2*I3**2*K11*K3-4*I1*I2**2*I4*K11*K3-4*I1**2*I2*I5*K11*K3-2*I1**2*I2**2*K12*K3-2*I2**4*K2*K3+3*I2*I3*I5**2*K1*K4-I1*I2*I3**2*K10*K4+4*I1*I2**2*I4*K10*K4+4*I1**2*I2*I5*K10*K4+I3**3*I4*K11*K4-3*I2*I3*I4**2*K11*K4-6*I1*I3*I4*I5*K11*K4+I1*I3**3*K12*K4-6*I1*I2*I3*I4*K12*K4-3*I1**2*I3*I5*K12*K4-3*I1**2*I2*I3*K13*K4+3*I2**2*I3*I5*K2*K4+I2**3*I3*K3*K4-3*I2**2*I4*I5*K4**2-3*I1*I2*I5**2*K4**2+3*I2**2*I3*I5*K1*K5+2*I1**2*I2**2*K10*K5+I1*I3**3*K11*K5-6*I1*I2*I3*I4*K11*K5-3*I1**2*I3*I5*K11*K5-3*I1**2*I2*I3*K12*K5+I2**3*I3*K2*K5-2*I2**3*I4*K4*K5-6*I1*I2**2*I5*K4*K5-I1*I2**3*K5**2-I3**2*I5**2*K1*K6+6*I2*I4*I5**2*K1*K6+2*I1*I5**3*K1*K6+2*I1*I2*I3*I4*K10*K6+I1**2*I3*I5*K10*K6-I3**2*I4**2*K11*K6+2*I2*I4**3*K11*K6+6*I1*I4**2*I5*K11*K6-2*I1*I3**2*I4*K12*K6+6*I1*I2*I4**2*K12*K6+6*I1**2*I4*I5*K12*K6-I1**2*I3**2*K13*K6+6*I1**2*I2*I4*K13*K6+2*I1**3*I5*K13*K6-2*I2*I3**2*I5*K2*K6+6*I2**2*I4*I5*K2*K6+6*I1*I2*I5**2*K2*K6-I2**2*I3**2*K3*K6+2*I2**3*I4*K3*K6+6*I1*I2**2*I5*K3*K6+2*I2*I3*I4*I5*K4*K6+I1*I3*I5**2*K4*K6+I2**2*I3*I4*K5*K6+2*I1*I2*I3*I5*K5*K6-2*I2*I4**2*I5*K6**2-2*I1*I4*I5**2*K6**2-2*I2*I3**2*I5*K1*K7+6*I2**2*I4*I5*K1*K7+6*I1*I2*I5**2*K1*K7+I1**2*I2*I3*K10*K7-2*I1*I3**2*I4*K11*K7+6*I1*I2*I4**2*K11*K7+6*I1**2*I4*I5*K11*K7-I1**2*I3**2*K12*K7+6*I1**2*I2*I4*K12*K7+2*I1**3*I5*K12*K7+2*I1**3*I2*K13*K7-I2**2*I3**2*K2*K7+2*I2**3*I4*K2*K7+6*I1*I2**2*I5*K2*K7+2*I1*I2**3*K3*K7+I2**2*I3*I4*K4*K7+2*I1*I2*I3*I5*K4*K7+I1*I2**2*I3*K5*K7-2*I2**2*I4**2*K6*K7-8*I1*I2*I4*I5*K6*K7-2*I1**2*I5**2*K6*K7-2*I1*I2**2*I4*K7**2-2*I1**2*I2*I5*K7**2-I2**2*I3**2*K1*K8+2*I2**3*I4*K1*K8+6*I1*I2**2*I5*K1*K8-I1**2*I3**2*K11*K8+6*I1**2*I2*I4*K11*K8+2*I1**3*I5*K11*K8+2*I1**3*I2*K12*K8+2*I1*I2**3*K2*K8+I1*I2**2*I3*K4*K8-4*I1*I2**2*I4*K6*K8-4*I1**2*I2*I5*K6*K8-2*I1**2*I2**2*K7*K8+I3**3*I5*K1*K9-6*I2*I3*I4*I5*K1*K9-3*I1*I3*I5**2*K1*K9-6*I1**2*I2*I4*K10*K9-2*I1**3*I5*K10*K9+3*I1*I3*I4**2*K11*K9+3*I1**2*I3*I4*K12*K9+I1**3*I3*K13*K9+I2*I3**3*K2*K9-3*I2**2*I3*I4*K2*K9-6*I1*I2*I3*I5*K2*K9-3*I1*I2**2*I3*K3*K9-I2*I3**2*I4*K4*K9+2*I2**2*I4**2*K4*K9-I1*I3**2*I5*K4*K9+8*I1*I2*I4*I5*K4*K9+2*I1**2*I5**2*K4*K9-I1*I2*I3**2*K5*K9+4*I1*I2**2*I4*K5*K9+4*I1**2*I2*I5*K5*K9+I2*I3*I4**2*K6*K9+2*I1*I3*I4*I5*K6*K9+2*I1*I2*I3*I4*K7*K9+I1**2*I3*I5*K7*K9+I1**2*I2*I3*K8*K9-3*I1*I2*I4**2*K9**2-3*I1**2*I4*I5*K9**2)
            P8 = (-I5**4*K1**2+I3**3*I5*K1*K10-6*I2*I3*I4*I5*K1*K10-3*I1*I3*I5**2*K1*K10-3*I1**2*I2*I4*K10**2-I1**3*I5*K10**2-2*I4**2*I5**2*K1*K11+3*I1*I3*I4**2*K10*K11-I4**4*K11**2+4*I3**2*I4*I5*K1*K12-4*I2*I4**2*I5*K1*K12-4*I1*I4*I5**2*K1*K12+3*I1**2*I3*I4*K10*K12-8*I1*I4**3*K11*K12-6*I1**2*I4**2*K12**2-I3**4*K1*K13+4*I2*I3**2*I4*K1*K13-2*I2**2*I4**2*K1*K13+4*I1*I3**2*I5*K1*K13-8*I1*I2*I4*I5*K1*K13-2*I1**2*I5**2*K1*K13+I1**3*I3*K10*K13-12*I1**2*I4**2*K11*K13-8*I1**3*I4*K12*K13-I1**4*K13**2-8*I2*I5**3*K1*K2+I2*I3**3*K10*K2-3*I2**2*I3*I4*K10*K2-6*I1*I2*I3*I5*K10*K2+4*I3**2*I4*I5*K11*K2-4*I2*I4**2*I5*K11*K2-4*I1*I4*I5**2*K11*K2-I3**4*K12*K2+4*I2*I3**2*I4*K12*K2-2*I2**2*I4**2*K12*K2+4*I1*I3**2*I5*K12*K2-8*I1*I2*I4*I5*K12*K2-2*I1**2*I5**2*K12*K2+4*I1*I2*I3**2*K13*K2-4*I1*I2**2*I4*K13*K2-4*I1**2*I2*I5*K13*K2-6*I2**2*I5**2*K2**2-12*I2**2*I5**2*K1*K3-3*I1*I2**2*I3*K10*K3-I3**4*K11*K3+4*I2*I3**2*I4*K11*K3-2*I2**2*I4**2*K11*K3+4*I1*I3**2*I5*K11*K3-8*I1*I2*I4*I5*K11*K3-2*I1**2*I5**2*K11*K3+4*I1*I2*I3**2*K12*K3-4*I1*I2**2*I4*K12*K3-4*I1**2*I2*I5*K12*K3-2*I1**2*I2**2*K13*K3-8*I2**3*I5*K2*K3-I2**4*K3**2+I3*I5**3*K1*K4-I2*I3**2*I4*K10*K4+2*I2**2*I4**2*K10*K4-I1*I3**2*I5*K10*K4+8*I1*I2*I4*I5*K10*K4+2*I1**2*I5**2*K10*K4-3*I3*I4**2*I5*K11*K4+I3**3*I4*K12*K4-3*I2*I3*I4**2*K12*K4-6*I1*I3*I4*I5*K12*K4+I1*I3**3*K13*K4-6*I1*I2*I3*I4*K13*K4-3*I1**2*I3*I5*K13*K4+3*I2*I3*I5**2*K2*K4+3*I2**2*I3*I5*K3*K4-3*I2*I4*I5**2*K4**2-I1*I5**3*K4**2+3*I2*I3*I5**2*K1*K5-I1*I2*I3**2*K10*K5+4*I1*I2**2*I4*K10*K5+4*I1**2*I2*I5*K10*K5+I3**3*I4*K11*K5-3*I2*I3*I4**2*K11*K5-6*I1*I3*I4*I5*K11*K5+I1*I3**3*K12*K5-6*I1*I2*I3*I4*K12*K5-3*I1**2*I3*I5*K12*K5-3*I1**2*I2*I3*K13*K5+3*I2**2*I3*I5*K2*K5+I2**3*I3*K3*K5-6*I2**2*I4*I5*K4*K5-6*I1*I2*I5**2*K4*K5-I2**3*I4*K5**2-3*I1*I2**2*I5*K5**2+2*I4*I5**3*K1*K6+I2*I3*I4**2*K10*K6+2*I1*I3*I4*I5*K10*K6+2*I4**3*I5*K11*K6-I3**2*I4**2*K12*K6+2*I2*I4**3*K12*K6+6*I1*I4**2*I5*K12*K6-2*I1*I3**2*I4*K13*K6+6*I1*I2*I4**2*K13*K6+6*I1**2*I4*I5*K13*K6-I3**2*I5**2*K2*K6+6*I2*I4*I5**2*K2*K6+2*I1*I5**3*K2*K6-2*I2*I3**2*I5*K3*K6+6*I2**2*I4*I5*K3*K6+6*I1*I2*I5**2*K3*K6+I3*I4*I5**2*K4*K6+2*I2*I3*I4*I5*K5*K6+I1*I3*I5**2*K5*K6-I4**2*I5**2*K6**2-I3**2*I5**2*K1*K7+6*I2*I4*I5**2*K1*K7+2*I1*I5**3*K1*K7+2*I1*I2*I3*I4*K10*K7+I1**2*I3*I5*K10*K7-I3**2*I4**2*K11*K7+2*I2*I4**3*K11*K7+6*I1*I4**2*I5*K11*K7-2*I1*I3**2*I4*K12*K7+6*I1*I2*I4**2*K12*K7+6*I1**2*I4*I5*K12*K7-I1**2*I3**2*K13*K7+6*I1**2*I2*I4*K13*K7+2*I1**3*I5*K13*K7-2*I2*I3**2*I5*K2*K7+6*I2**2*I4*I5*K2*K7+6*I1*I2*I5**2*K2*K7-I2**2*I3**2*K3*K7+2*I2**3*I4*K3*K7+6*I1*I2**2*I5*K3*K7+2*I2*I3*I4*I5*K4*K7+I1*I3*I5**2*K4*K7+I2**2*I3*I4*K5*K7+2*I1*I2*I3*I5*K5*K7-4*I2*I4**2*I5*K6*K7-4*I1*I4*I5**2*K6*K7-I2**2*I4**2*K7**2-4*I1*I2*I4*I5*K7**2-I1**2*I5**2*K7**2-2*I2*I3**2*I5*K1*K8+6*I2**2*I4*I5*K1*K8+6*I1*I2*I5**2*K1*K8+I1**2*I2*I3*K10*K8-2*I1*I3**2*I4*K11*K8+6*I1*I2*I4**2*K11*K8+6*I1**2*I4*I5*K11*K8-I1**2*I3**2*K12*K8+6*I1**2*I2*I4*K12*K8+2*I1**3*I5*K12*K8+2*I1**3*I2*K13*K8-I2**2*I3**2*K2*K8+2*I2**3*I4*K2*K8+6*I1*I2**2*I5*K2*K8+2*I1*I2**3*K3*K8+I2**2*I3*I4*K4*K8+2*I1*I2*I3*I5*K4*K8+I1*I2**2*I3*K5*K8-2*I2**2*I4**2*K6*K8-8*I1*I2*I4*I5*K6*K8-2*I1**2*I5**2*K6*K8-4*I1*I2**2*I4*K7*K8-4*I1**2*I2*I5*K7*K8-I1**2*I2**2*K8**2-3*I3*I4*I5**2*K1*K9-6*I1*I2*I4**2*K10*K9-6*I1**2*I4*I5*K10*K9+I3*I4**3*K11*K9+3*I1*I3*I4**2*K12*K9+3*I1**2*I3*I4*K13*K9+I3**3*I5*K2*K9-6*I2*I3*I4*I5*K2*K9-3*I1*I3*I5**2*K2*K9+I2*I3**3*K3*K9-3*I2**2*I3*I4*K3*K9-6*I1*I2*I3*I5*K3*K9-I3**2*I4*I5*K4*K9+4*I2*I4**2*I5*K4*K9+4*I1*I4*I5**2*K4*K9-I2*I3**2*I4*K5*K9+2*I2**2*I4**2*K5*K9-I1*I3**2*I5*K5*K9+8*I1*I2*I4*I5*K5*K9+2*I1**2*I5**2*K5*K9+I3*I4**2*I5*K6*K9+I2*I3*I4**2*K7*K9+2*I1*I3*I4*I5*K7*K9+2*I1*I2*I3*I4*K8*K9+I1**2*I3*I5*K8*K9-I2*I4**3*K9**2-3*I1*I4**2*I5*K9**2)
            P6 = (-3*I3*I4*I5**2*K1*K10-3*I1*I2*I4**2*K10**2-3*I1**2*I4*I5*K10**2+I3*I4**3*K10*K11-2*I4**2*I5**2*K1*K12+3*I1*I3*I4**2*K10*K12-2*I4**4*K11*K12-4*I1*I4**3*K12**2+4*I3**2*I4*I5*K1*K13-4*I2*I4**2*I5*K1*K13-4*I1*I4*I5**2*K1*K13+3*I1**2*I3*I4*K10*K13-8*I1*I4**3*K11*K13-12*I1**2*I4**2*K12*K13-4*I1**3*I4*K13**2-2*I5**4*K1*K2+I3**3*I5*K10*K2-6*I2*I3*I4*I5*K10*K2-3*I1*I3*I5**2*K10*K2-2*I4**2*I5**2*K11*K2+4*I3**2*I4*I5*K12*K2-4*I2*I4**2*I5*K12*K2-4*I1*I4*I5**2*K12*K2-I3**4*K13*K2+4*I2*I3**2*I4*K13*K2-2*I2**2*I4**2*K13*K2+4*I1*I3**2*I5*K13*K2-8*I1*I2*I4*I5*K13*K2-2*I1**2*I5**2*K13*K2-4*I2*I5**3*K2**2-8*I2*I5**3*K1*K3+I2*I3**3*K10*K3-3*I2**2*I3*I4*K10*K3-6*I1*I2*I3*I5*K10*K3+4*I3**2*I4*I5*K11*K3-4*I2*I4**2*I5*K11*K3-4*I1*I4*I5**2*K11*K3-I3**4*K12*K3+4*I2*I3**2*I4*K12*K3-2*I2**2*I4**2*K12*K3+4*I1*I3**2*I5*K12*K3-8*I1*I2*I4*I5*K12*K3-2*I1**2*I5**2*K12*K3+4*I1*I2*I3**2*K13*K3-4*I1*I2**2*I4*K13*K3-4*I1**2*I2*I5*K13*K3-12*I2**2*I5**2*K2*K3-4*I2**3*I5*K3**2-I3**2*I4*I5*K10*K4+4*I2*I4**2*I5*K10*K4+4*I1*I4*I5**2*K10*K4-3*I3*I4**2*I5*K12*K4+I3**3*I4*K13*K4-3*I2*I3*I4**2*K13*K4-6*I1*I3*I4*I5*K13*K4+I3*I5**3*K2*K4+3*I2*I3*I5**2*K3*K4-I4*I5**3*K4**2+I3*I5**3*K1*K5-I2*I3**2*I4*K10*K5+2*I2**2*I4**2*K10*K5-I1*I3**2*I5*K10*K5+8*I1*I2*I4*I5*K10*K5+2*I1**2*I5**2*K10*K5-3*I3*I4**2*I5*K11*K5+I3**3*I4*K12*K5-3*I2*I3*I4**2*K12*K5-6*I1*I3*I4*I5*K12*K5+I1*I3**3*K13*K5-6*I1*I2*I3*I4*K13*K5-3*I1**2*I3*I5*K13*K5+3*I2*I3*I5**2*K2*K5+3*I2**2*I3*I5*K3*K5-6*I2*I4*I5**2*K4*K5-2*I1*I5**3*K4*K5-3*I2**2*I4*I5*K5**2-3*I1*I2*I5**2*K5**2+I3*I4**2*I5*K10*K6+2*I4**3*I5*K12*K6-I3**2*I4**2*K13*K6+2*I2*I4**3*K13*K6+6*I1*I4**2*I5*K13*K6+2*I4*I5**3*K2*K6-I3**2*I5**2*K3*K6+6*I2*I4*I5**2*K3*K6+2*I1*I5**3*K3*K6+I3*I4*I5**2*K5*K6+2*I4*I5**3*K1*K7+I2*I3*I4**2*K10*K7+2*I1*I3*I4*I5*K10*K7+2*I4**3*I5*K11*K7-I3**2*I4**2*K12*K7+2*I2*I4**3*K12*K7+6*I1*I4**2*I5*K12*K7-2*I1*I3**2*I4*K13*K7+6*I1*I2*I4**2*K13*K7+6*I1**2*I4*I5*K13*K7-I3**2*I5**2*K2*K7+6*I2*I4*I5**2*K2*K7+2*I1*I5**3*K2*K7-2*I2*I3**2*I5*K3*K7+6*I2**2*I4*I5*K3*K7+6*I1*I2*I5**2*K3*K7+I3*I4*I5**2*K4*K7+2*I2*I3*I4*I5*K5*K7+I1*I3*I5**2*K5*K7-2*I4**2*I5**2*K6*K7-2*I2*I4**2*I5*K7**2-2*I1*I4*I5**2*K7**2-I3**2*I5**2*K1*K8+6*I2*I4*I5**2*K1*K8+2*I1*I5**3*K1*K8+2*I1*I2*I3*I4*K10*K8+I1**2*I3*I5*K10*K8-I3**2*I4**2*K11*K8+2*I2*I4**3*K11*K8+6*I1*I4**2*I5*K11*K8-2*I1*I3**2*I4*K12*K8+6*I1*I2*I4**2*K12*K8+6*I1**2*I4*I5*K12*K8-I1**2*I3**2*K13*K8+6*I1**2*I2*I4*K13*K8+2*I1**3*I5*K13*K8-2*I2*I3**2*I5*K2*K8+6*I2**2*I4*I5*K2*K8+6*I1*I2*I5**2*K2*K8-I2**2*I3**2*K3*K8+2*I2**3*I4*K3*K8+6*I1*I2**2*I5*K3*K8+2*I2*I3*I4*I5*K4*K8+I1*I3*I5**2*K4*K8+I2**2*I3*I4*K5*K8+2*I1*I2*I3*I5*K5*K8-4*I2*I4**2*I5*K6*K8-4*I1*I4*I5**2*K6*K8-2*I2**2*I4**2*K7*K8-8*I1*I2*I4*I5*K7*K8-2*I1**2*I5**2*K7*K8-2*I1*I2**2*I4*K8**2-2*I1**2*I2*I5*K8**2-2*I2*I4**3*K10*K9-6*I1*I4**2*I5*K10*K9+I3*I4**3*K12*K9+3*I1*I3*I4**2*K13*K9-3*I3*I4*I5**2*K2*K9+I3**3*I5*K3*K9-6*I2*I3*I4*I5*K3*K9-3*I1*I3*I5**2*K3*K9+2*I4**2*I5**2*K4*K9-I3**2*I4*I5*K5*K9+4*I2*I4**2*I5*K5*K9+4*I1*I4*I5**2*K5*K9+I3*I4**2*I5*K7*K9+I2*I3*I4**2*K8*K9+2*I1*I3*I4*I5*K8*K9-I4**3*I5*K9**2)
            P4 = (-I2*I4**3*K10**2-3*I1*I4**2*I5*K10**2+I3*I4**3*K10*K12-I4**4*K12**2-2*I4**2*I5**2*K1*K13+3*I1*I3*I4**2*K10*K13-2*I4**4*K11*K13-8*I1*I4**3*K12*K13-6*I1**2*I4**2*K13**2-3*I3*I4*I5**2*K10*K2-2*I4**2*I5**2*K12*K2+4*I3**2*I4*I5*K13*K2-4*I2*I4**2*I5*K13*K2-4*I1*I4*I5**2*K13*K2-I5**4*K2**2-2*I5**4*K1*K3+I3**3*I5*K10*K3-6*I2*I3*I4*I5*K10*K3-3*I1*I3*I5**2*K10*K3-2*I4**2*I5**2*K11*K3+4*I3**2*I4*I5*K12*K3-4*I2*I4**2*I5*K12*K3-4*I1*I4*I5**2*K12*K3-I3**4*K13*K3+4*I2*I3**2*I4*K13*K3-2*I2**2*I4**2*K13*K3+4*I1*I3**2*I5*K13*K3-8*I1*I2*I4*I5*K13*K3-2*I1**2*I5**2*K13*K3-8*I2*I5**3*K2*K3-6*I2**2*I5**2*K3**2+2*I4**2*I5**2*K10*K4-3*I3*I4**2*I5*K13*K4+I3*I5**3*K3*K4-I3**2*I4*I5*K10*K5+4*I2*I4**2*I5*K10*K5+4*I1*I4*I5**2*K10*K5-3*I3*I4**2*I5*K12*K5+I3**3*I4*K13*K5-3*I2*I3*I4**2*K13*K5-6*I1*I3*I4*I5*K13*K5+I3*I5**3*K2*K5+3*I2*I3*I5**2*K3*K5-2*I4*I5**3*K4*K5-3*I2*I4*I5**2*K5**2-I1*I5**3*K5**2+2*I4**3*I5*K13*K6+2*I4*I5**3*K3*K6+I3*I4**2*I5*K10*K7+2*I4**3*I5*K12*K7-I3**2*I4**2*K13*K7+2*I2*I4**3*K13*K7+6*I1*I4**2*I5*K13*K7+2*I4*I5**3*K2*K7-I3**2*I5**2*K3*K7+6*I2*I4*I5**2*K3*K7+2*I1*I5**3*K3*K7+I3*I4*I5**2*K5*K7-I4**2*I5**2*K7**2+2*I4*I5**3*K1*K8+I2*I3*I4**2*K10*K8+2*I1*I3*I4*I5*K10*K8+2*I4**3*I5*K11*K8-I3**2*I4**2*K12*K8+2*I2*I4**3*K12*K8+6*I1*I4**2*I5*K12*K8-2*I1*I3**2*I4*K13*K8+6*I1*I2*I4**2*K13*K8+6*I1**2*I4*I5*K13*K8-I3**2*I5**2*K2*K8+6*I2*I4*I5**2*K2*K8+2*I1*I5**3*K2*K8-2*I2*I3**2*I5*K3*K8+6*I2**2*I4*I5*K3*K8+6*I1*I2*I5**2*K3*K8+I3*I4*I5**2*K4*K8+2*I2*I3*I4*I5*K5*K8+I1*I3*I5**2*K5*K8-2*I4**2*I5**2*K6*K8-4*I2*I4**2*I5*K7*K8-4*I1*I4*I5**2*K7*K8-I2**2*I4**2*K8**2-4*I1*I2*I4*I5*K8**2-I1**2*I5**2*K8**2-2*I4**3*I5*K10*K9+I3*I4**3*K13*K9-3*I3*I4*I5**2*K3*K9+2*I4**2*I5**2*K5*K9+I3*I4**2*I5*K8*K9)
            P2 = (-I4**3*I5*K10**2+I3*I4**3*K10*K13-2*I4**4*K12*K13-4*I1*I4**3*K13**2-2*I4**2*I5**2*K13*K2-3*I3*I4*I5**2*K10*K3-2*I4**2*I5**2*K12*K3+4*I3**2*I4*I5*K13*K3-4*I2*I4**2*I5*K13*K3-4*I1*I4*I5**2*K13*K3-2*I5**4*K2*K3-4*I2*I5**3*K3**2+2*I4**2*I5**2*K10*K5-3*I3*I4**2*I5*K13*K5+I3*I5**3*K3*K5-I4*I5**3*K5**2+2*I4**3*I5*K13*K7+2*I4*I5**3*K3*K7+I3*I4**2*I5*K10*K8+2*I4**3*I5*K12*K8-I3**2*I4**2*K13*K8+2*I2*I4**3*K13*K8+6*I1*I4**2*I5*K13*K8+2*I4*I5**3*K2*K8-I3**2*I5**2*K3*K8+6*I2*I4*I5**2*K3*K8+2*I1*I5**3*K3*K8+I3*I4*I5**2*K5*K8-2*I4**2*I5**2*K7*K8-2*I2*I4**2*I5*K8**2-2*I1*I4*I5**2*K8**2)
            P0 = (-I4**4*K13**2-2*I4**2*I5**2*K13*K3-I5**4*K3**2+2*I4**3*I5*K13*K8+2*I4*I5**3*K3*K8-I4**2*I5**2*K8**2)
            P = [P0, P2, P4, P6, P8, P10, P12, P14, P16]
            G = [G1, G2, G3, G4, G5]
            H = [H1, H2, H3, H4, H5]
            I = [I1, I2, I3, I4, I5]
            return (P, G, H, I)
        (self.P, self.G, self.H, self.I) = calc_parameters(B1X, B1Y, B2X, B2Y, B3X, B3Y, psi1, psi2, psi3, l1, l2, l3, t1, t2, t3)
        P0, P2, P4, P6, P8, P10, P12, P14, P16 = self.P[0], self.P[1], self.P[2], self.P[3], self.P[4], self.P[5], self.P[6], self.P[7], self.P[8]
        G1, G2, G3, G4, G5 = self.G[0], self.G[1], self.G[2], self.G[3], self.G[4]
        H1, H2, H3, H4, H5 = self.H[0], self.H[1], self.H[2], self.H[3], self.H[4]
        I1, I2, I3, I4, I5 = self.I[0], self.I[1], self.I[2], self.I[3], self.I[4]
        # 'Unpacks' parameters needed to find the other two angles.
        p=[P16, P14, P12, P10, P8, P6, P4, P2, P0]
        normp = np.array(p)
        normp /= np.abs(normp).max()
        # Normalises the polynomial coefficients to make the solution process run faster.
        solutions = np.roots(normp)
        sols = [np.sqrt(i) for i in solutions if i > 0]
        xvals = []
        theta1, theta2, theta3 = 0, 0, 0
        for i in range(len(sols)):
            if 0.2679 < np.real(sols[i]) < 0.57735:
                # Selects the roots in the correct range.
                xvals.append(np.real(sols[i]))
        for i in range(len(xvals)):
            phi1 = 2*np.arctan(xvals[i])
            theta1a = ((np.pi/2)-phi1)
            A2 = G1*xvals[i]**2+G2
            B2 = G3*xvals[i]
            C2 = G4*xvals[i]**2+G5
            if (B2**2-4*A2*C2) < 0:
                # If (B2**2-4*A2*C2) < 0 this means that x2 is complex and there are no solutions.
                print 'There are no solutions for this tripod position - DO NOT MOVE TO THIS POSITION'
                quit()
            x2 = (-B2+(B2**2-4*A2*C2)**0.5)/(2*A2)
            phi2 = 2*np.arctan(x2)
            theta2a = ((np.pi/2)-phi2)
            A3 = I1*xvals[i]**2+I4
            B3 = I3*xvals[i]
            C3 = I2*xvals[i]**2+I5
            if (B3**2-4*A3*C3) < 0:
                # If (B3**2-4*A3*C3) < 0 this means that x3 is complex and there are no solutions.
                print 'There are no solutions for this tripod position - DO NOT MOVE TO THIS POSITION'
                quit()
            x3 = (-B3-(B3**2-4*A3*C3)**0.5)/(2*A3)
            phi3 = 2*np.arctan(x3)
            theta3a = ((np.pi/2)-phi3)
            test = (H1*x3**2+H4)*x2**2+H3*x3*x2+(H2*x3**2+H5)
            if abs(test) < 0.1:
                theta1, theta2, theta3 = theta1a, theta2a, theta3a
        if theta1 == 0:
            # If theta1 has not been reset the program has not found any solutions.
            print 'There are no solutions for this tripod position - DO NOT MOVE TO THIS POSITION'
            quit()
        elif theta1 != 0:
            pass
        self.theta[0], self.theta[1], self.theta[2] = theta1, theta2, theta3
        # can remove - for debug
        return (theta1, theta2, theta3)

    def _calcBXY(self, C, (alpha1, alpha2, alpha3)):
        # Calculate base translations (see documentation).
        (xvec, yvec, zvec) = self._calc_xyzvec(alpha1, alpha2, alpha3)
        # Calculate T1,2,3 from xvec, yvec, zvec.
        T2 = C-(self.c[0]*xvec+self.c[1]*yvec+self.c[2]*zvec)
        T1 = T2+self.t[2]*yvec
        cos_t2 = (self.t[0]**2+self.t[2]**2-self.t[1]**2)/(2*self.t[0]*self.t[2])  # Cosine rule.
        sin_t2 = np.sqrt(1-cos_t2**2)
        T3 = T2 + xvec*self.t[0]*sin_t2 + yvec*self.t[0]*cos_t2
        # Calc B from T.
        cos_theta1 = T1[2]/self.l[0]  # T1[2]==T1.z etc
        cos_theta2 = T2[2]/self.l[1]
        cos_theta3 = T3[2]/self.l[2]
        theta1 = np.arccos(cos_theta1)
        theta2 = np.arccos(cos_theta2)
        theta3 = -np.arccos(cos_theta3)
        self.theta[0], self.theta[1], self.theta[2] = theta1, theta2, theta3
        sin_theta1 = np.sqrt(1-cos_theta1**2)*np.sign(self.theta[0])
        sin_theta2 = np.sqrt(1-cos_theta2**2)*np.sign(self.theta[1])
        sin_theta3 = np.sqrt(1-cos_theta3**2)*np.sign(self.theta[2])
        v1 = np.array([np.cos(self.psi[0])*sin_theta1, np.sin(self.psi[0])*sin_theta1, cos_theta1])
        v2 = np.array([np.cos(self.psi[1])*sin_theta2, np.sin(self.psi[1])*sin_theta2, cos_theta2])
        v3 = np.array([np.cos(self.psi[2])*sin_theta3, np.sin(self.psi[2])*sin_theta3, cos_theta3])
        B1, B2, B3 = T1-v1*self.l[0], T2-v2*self.l[1], T3-v3*self.l[2]
        X = [B1[0]-self.BX[0], B2[0]-self.BX[1], B3[0]-self.BX[2]]
        Y = [B1[1]-self.BY[0], B2[1]-self.BY[1], B3[1]-self.BY[2]]
        # X[leg1,2,3], B1,2,3[X,Y,Z], BX[leg1,2,3]
        return X, Y

    def _calc_xyzvec(self, alpha1, alpha2, alpha3):
        # Calculate top plate coordinate vectors (see documentation).
        xvec = np.array([np.cos(alpha2)*np.cos(alpha3), np.cos(alpha2)*np.sin(alpha3), -np.sin(alpha2)])
        yvec = np.array([np.sin(alpha1)*np.sin(alpha2)*np.cos(alpha3)-np.cos(alpha1)*np.sin(alpha3), np.sin(alpha1)*np.sin(alpha2)*np.sin(alpha3)+np.cos(alpha1)*np.cos(alpha3), np.sin(alpha1)*np.cos(alpha2)])
        zvec = np.array([np.cos(alpha1)*np.sin(alpha2)*np.cos(alpha3)+np.sin(alpha1)*np.sin(alpha3), np.cos(alpha1)*np.sin(alpha2)*np.sin(alpha3)-np.sin(alpha1)*np.cos(alpha3), np.cos(alpha1)*np.cos(alpha2)])
        return (xvec, yvec, zvec)

    def cbase(self, C, (alpha1, alpha2, alpha3)):
        '''self.cbase((CX, CY, CZ),(alpha1,alpha2, alpha3)) calculates base vectors
        for tooling point coordinates and angles (degrees)'''
        X, Y = self._calcBXY(C, (alpha1*pi/180, alpha2*pi/180, alpha3*pi/180))
        self.X, self.Y = X, Y
        return np.array(X), np.array(Y)

tp = tripod_class()

tp.ctool([0, 0, 0], [0, 0, 0])
print tp


print 'Test cases: calculate angles and tool point coordinates for particular slide translations, then carry out the reverse calculation to check that the translations are the original values'
print '\nFirst a 1mm translation of one slide\n'
X, Y = [0, 0, 1.0], [0, 0, 0]
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
# calculate tool-point coordinates c and tilt angles alpha for given X and Y
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

print '\nNow we move the tripod stage upwards (in the Z direction) by 5mm and return it to its original position as a further check'

X, Y = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
C, alpha = tp.ctool(X, Y)
# calculate tool-point coordinates C and tilt angles alpha for given X Y base translations
C[2] += 5.0
# set this to be C[0] to move in the X-direction, C[1] to move in the Y-direction and C[2] to move in the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

C[2] -= 5.0
# set this to be C[0] to move in the X-direction, C[1] to move in the Y-direction and C[2] to move in the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

print '\nNext we perform a rotation about the Y axis and then rotate the tripod back'

X, Y = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
C, alpha = tp.ctool(X, Y)
# calculate tool-point coordinates C and tilt angles alpha for given X Y base translations
alpha[1] += 0.5
# set this to be alpha[0] to rotate about the X-direction, alpha[1] to rotate about the Y-direction and alpha[2] to rotate about the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

alpha[1] -= 0.5
# set this to be alpha[0] to rotate about the X-direction, alpha[1] to rotate about the Y-direction and alpha[2] to rotate about the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

print '\nAnd finally a rotation about the Z axis'

X, Y = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
C, alpha = tp.ctool(X, Y)
# calculate tool-point coordinates C and tilt angles alpha for given X Y base translations
alpha[2] += 0.5
# set this to be alpha[0] to rotate about the X-direction, alpha[1] to rotate about the Y-direction and alpha[2] to rotate about the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)

alpha[2] -= 0.5
# set this to be alpha[0] to rotate about the X-direction, alpha[1] to rotate about the Y-direction and alpha[2] to rotate about the Z-direction
X, Y = tp.cbase(C, np.array(alpha))
print '\nX coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
C, alpha = tp.ctool(X, Y)
print 'Tool point coords (C):  %.5f %.5f %.5f' % tuple(C)
print 'Tilt angles (alpha, deg) :%.5f %.5f %.5f' % tuple(alpha)
print 'Reverse calculation to check that X and Y are reproduced'
X, Y = tp.cbase(C, np.array(alpha))
print 'X coords of base translations applied: %.5f %.5f %.5f' % tuple(X)
print 'Y coords of base translations applied: %.5f %.5f %.5f' % tuple(Y)
