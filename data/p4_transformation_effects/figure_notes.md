# P4 transformation effect figures

Generated from saved Z=0 real-ray coordinates, the saved all-grid radial fit, saved paraxial scale metrics, and saved v2 joint-fit K parameters. No CODE V run or fit was performed during plotting; the saved K parameters were not refit.

Power law: k1(A)=c1*A^n+c0 with c0=-0.0184792556675 mm^-2, c1=0.000180236991874 mm^-2/D^n, n=1.53545703384. A=0 baseline inverse uses fitted c0 and then reapplies c0, enforcing identity at A=0; inverse radial radius is solved on the monotonic near-axis branch. Baseline roundtrip maximum error=2.22e-13 um.
Barrel composition is B_A = D_k(A) o S_s(A) o D_k(0)^-1: invert the fitted physical radial law at A=0, apply the saved E=0 paraxial scale ratio s(A), then apply the power-law physical k1(A). At A=5 D, s=0.896199673581, k1(A)=-0.0163458063013 mm^-2; at A=0, s=1, k1(A)=-0.0184792556675 mm^-2.
At A=5 D, new barrel versus old joint-fit beta barrel baseline: corner RMS/max difference=0.051555/0.051555 um; all-nine RMS/max difference=42.030309/63.045442 um. Old beta=-0.0429471823923 normalized; its corner calibration is retained through the previously fitted sx, sy, q, applied after the new barrel map.

A=5 D, E=0 deg barrel prediction versus actual:
New power-law barrel: corners RMS/max=0.096530/0.096530 um; five other points RMS/max=1.008722/1.127786 um; all-nine RMS/max=0.754606/1.127786 um.
Old beta barrel: corners RMS/max=0.044975/0.044975 um; five other points RMS/max=55.380836/61.917657 um; all-nine RMS/max=41.278449/61.917657 um.

A=5 D, E=20 deg joint prediction versus actual:
New power barrel + saved K: corners RMS/max=19.501095/21.320110 um; five other points RMS/max=18.651834/26.438233 um; all-nine RMS/max=19.033962/26.438233 um.
Old beta barrel + saved K: corners RMS/max=19.493973/21.306156 um; five other points RMS/max=60.208723/88.634173 um; all-nine RMS/max=46.720815/88.634173 um.

The four corners were used for the saved K fit; the other five native field points are checks. In the sweep figure, circles mark actual samples and squares mark predicted samples at all nine native field locations. Every keystone panel plots x=physical y and y=physical x. The isolated keystone panel applies saved K to the unchanged reference and is not a standalone optical state. Each trace shows the outer perimeter lines and its center sample as a dot; the middle native row and column are omitted.
