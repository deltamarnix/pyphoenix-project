# generated file
from flopy4.array import MFArray
from flopy4.compound import MFRecord, MFList
from flopy4.package import MFPackage
from flopy4.scalar import MFDouble, MFFilename, MFInteger, MFKeyword, MFString


class GwfNpf(MFPackage):
    multipkg = False
    stress = False
    advanced = False

    save_flows = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to save NPF flows""",
        description =
"""keyword to indicate that budget flow terms will be written to the file
specified with ``BUDGET SAVE FILE'' in Output Control.""",
    )

    print_flows = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to print NPF flows to listing file""",
        description =
"""keyword to indicate that calculated flows between cells will be
printed to the listing file for every stress period time step in which
``BUDGET PRINT'' is specified in Output Control. If there is no Output
Control option and ``PRINT_FLOWS'' is specified, then flow rates are
printed for the last time step of each stress period.  This option can
produce extremely large list files because all cell-by-cell flows are
printed.  It should only be used with the NPF Package for models that
have a small number of cells.""",
    )

    alternative_cell_averaging = MFString(
        type = "string",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""conductance weighting option""",
        description =
"""is a text keyword to indicate that an alternative method will be used
for calculating the conductance for horizontal cell connections.  The
text value for ALTERNATIVE_CELL_AVERAGING can be ``LOGARITHMIC'',
``AMT-LMK'', or ``AMT-HMK''.  ``AMT-LMK'' signifies that the
conductance will be calculated using arithmetic-mean thickness and
logarithmic-mean hydraulic conductivity.  ``AMT-HMK'' signifies that
the conductance will be calculated using arithmetic-mean thickness and
harmonic-mean hydraulic conductivity. If the user does not specify a
value for ALTERNATIVE_CELL_AVERAGING, then the harmonic-mean method
will be used.  This option cannot be used if the XT3D option is
invoked.""",
    )

    thickstrt = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to activate THICKSTRT option""",
        description =
"""indicates that cells having a negative ICELLTYPE are confined, and
their cell thickness for conductance calculations will be computed as
STRT-BOT rather than TOP-BOT.  This option should be used with caution
as it only affects conductance calculations in the NPF Package.""",
    )

    cvoptions = MFRecord(
        type = "record",
        params = {
            "variablecv": MFKeyword(),
            "dewatered": MFKeyword(),
        },
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""vertical conductance options""",
        description =
"""none""",
    )

    variablecv = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""keyword to activate VARIABLECV option""",
        description =
"""keyword to indicate that the vertical conductance will be calculated
using the saturated thickness and properties of the overlying cell and
the thickness and properties of the underlying cell.  If the DEWATERED
keyword is also specified, then the vertical conductance is calculated
using only the saturated thickness and properties of the overlying
cell if the head in the underlying cell is below its top.  If these
keywords are not specified, then the default condition is to calculate
the vertical conductance at the start of the simulation using the
initial head and the cell properties.  The vertical conductance
remains constant for the entire simulation.""",
    )

    dewatered = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to activate DEWATERED option""",
        description =
"""If the DEWATERED keyword is specified, then the vertical conductance
is calculated using only the saturated thickness and properties of the
overlying cell if the head in the underlying cell is below its top.""",
    )

    perched = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to activate PERCHED option""",
        description =
"""keyword to indicate that when a cell is overlying a dewatered
convertible cell, the head difference used in Darcy's Law is equal to
the head in the overlying cell minus the bottom elevation of the
overlying cell.  If not specified, then the default is to use the head
difference between the two cells.""",
    )

    rewet_record = MFRecord(
        type = "record",
        params = {
            "rewet": MFKeyword(),
            "wetfct": MFDouble(),
            "iwetit": MFInteger(),
            "ihdwet": MFInteger(),
        },
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""""",
        description =
"""""",
    )

    rewet = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""keyword to activate rewetting""",
        description =
"""activates model rewetting.  Rewetting is off by default.""",
    )

    wetfct = MFDouble(
        type = "double",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""wetting factor to use for rewetting""",
        description =
"""is a keyword and factor that is included in the calculation of the
head that is initially established at a cell when that cell is
converted from dry to wet.""",
    )

    iwetit = MFInteger(
        type = "integer",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""interval to use for rewetting""",
        description =
"""is a keyword and iteration interval for attempting to wet cells.
Wetting is attempted every IWETIT iteration. This applies to outer
iterations and not inner iterations. If IWETIT is specified as zero or
less, then the value is changed to 1.""",
    )

    ihdwet = MFInteger(
        type = "integer",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""flag to determine wetting equation""",
        description =
"""is a keyword and integer flag that determines which equation is used
to define the initial head at cells that become wet.  If IHDWET is 0,
h = BOT + WETFCT (hm - BOT). If IHDWET is not 0, h = BOT + WETFCT
(THRESH).""",
    )

    xt3doptions = MFRecord(
        type = "record",
        params = {
            "xt3d": MFKeyword(),
            "rhs": MFKeyword(),
        },
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to activate XT3D""",
        description =
"""none""",
    )

    xt3d = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""keyword to activate XT3D""",
        description =
"""keyword indicating that the XT3D formulation will be used.  If the RHS
keyword is also included, then the XT3D additional terms will be added
to the right-hand side.  If the RHS keyword is excluded, then the XT3D
terms will be put into the coefficient matrix.  Use of XT3D will
substantially increase the computational effort, but will result in
improved accuracy for anisotropic conductivity fields and for
unstructured grids in which the CVFD requirement is violated.  XT3D
requires additional information about the shapes of grid cells.  If
XT3D is active and the DISU Package is used, then the user will need
to provide in the DISU Package the angldegx array in the
CONNECTIONDATA block and the VERTICES and CELL2D blocks.""",
    )

    rhs = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to XT3D on right hand side""",
        description =
"""If the RHS keyword is also included, then the XT3D additional terms
will be added to the right-hand side.  If the RHS keyword is excluded,
then the XT3D terms will be put into the coefficient matrix.""",
    )

    save_specific_discharge = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to save specific discharge""",
        description =
"""keyword to indicate that x, y, and z components of specific discharge
will be calculated at cell centers and written to the budget file,
which is specified with ``BUDGET SAVE FILE'' in Output Control.  If
this option is activated, then additional information may be required
in the discretization packages and the GWF Exchange package (if GWF
models are coupled).  Specifically, ANGLDEGX must be specified in the
CONNECTIONDATA block of the DISU Package; ANGLDEGX must also be
specified for the GWF Exchange as an auxiliary variable.""",
    )

    save_saturation = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to save saturation""",
        description =
"""keyword to indicate that cell saturation will be written to the budget
file, which is specified with ``BUDGET SAVE FILE'' in Output Control.
Saturation will be saved to the budget file as an auxiliary variable
saved with the DATA-SAT text label.  Saturation is a cell variable
that ranges from zero to one and can be used by post processing
programs to determine how much of a cell volume is saturated.  If
ICELLTYPE is 0, then saturation is always one.""",
    )

    k22overk = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to indicate that specified K22 is a ratio""",
        description =
"""keyword to indicate that specified K22 is a ratio of K22 divided by K.
If this option is specified, then the K22 array entered in the NPF
Package will be multiplied by K after being read.""",
    )

    k33overk = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""keyword to indicate that specified K33 is a ratio""",
        description =
"""keyword to indicate that specified K33 is a ratio of K33 divided by K.
If this option is specified, then the K33 array entered in the NPF
Package will be multiplied by K after being read.""",
    )

    tvk_filerecord = MFRecord(
        type = "record",
        params = {
            "tvk6": MFKeyword(),
            "filein": MFKeyword(),
            "tvk6_filename": MFString(),
        },
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""""",
        description =
"""""",
    )

    tvk6 = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""tvk keyword""",
        description =
"""keyword to specify that record corresponds to a time-varying hydraulic
conductivity (TVK) file.  The behavior of TVK and a description of the
input file is provided separately.""",
    )

    filein = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""file keyword""",
        description =
"""keyword to specify that an input filename is expected next.""",
    )

    tvk6_filename = MFString(
        type = "string",
        block = "options",
        shape = "",
        reader = "urword",
        optional = False,
        longname =
"""file name of TVK information""",
        description =
"""defines a time-varying hydraulic conductivity (TVK) input file.
Records in the TVK file can be used to change hydraulic conductivity
properties at specified times or stress periods.""",
    )

    export_array_ascii = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""export array variables to layered ascii files.""",
        description =
"""keyword that specifies input griddata arrays should be written to
layered ascii output files.""",
    )

    export_array_netcdf = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""export array variables to netcdf output files.""",
        description =
"""keyword that specifies input griddata arrays should be written to the
model output netcdf file.""",
    )

    dev_no_newton = MFKeyword(
        type = "keyword",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""turn off Newton for unconfined cells""",
        description =
"""turn off Newton for unconfined cells""",
    )

    dev_omega = MFDouble(
        type = "double",
        block = "options",
        shape = "",
        reader = "urword",
        optional = True,
        longname =
"""set saturation omega value""",
        description =
"""set saturation omega value""",
    )

    icelltype = MFArray(
        type = "integer",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = False,
        longname =
"""confined or convertible indicator""",
        description =
"""flag for each cell that specifies how saturated thickness is treated.
0 means saturated thickness is held constant;  $>$0 means saturated
thickness varies with computed head when head is below the cell top;
$<$0 means saturated thickness varies with computed head unless the
THICKSTRT option is in effect.  When THICKSTRT is in effect, a
negative value for ICELLTYPE indicates that the saturated thickness
value used in conductance calculations in the NPF Package will be
computed as STRT-BOT and held constant.  If the THICKSTRT option is
not in effect, then negative values provided by the user for ICELLTYPE
are automatically reassigned by the program to a value of one.""",
    )

    k = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = False,
        longname =
"""hydraulic conductivity (L/T)""",
        description =
"""is the hydraulic conductivity.  For the common case in which the user
would like to specify the horizontal hydraulic conductivity and the
vertical hydraulic conductivity, then K should be assigned as the
horizontal hydraulic conductivity, K33 should be assigned as the
vertical hydraulic conductivity, and K22 and the three rotation angles
should not be specified.  When more sophisticated anisotropy is
required, then K corresponds to the K11 hydraulic conductivity axis.
All included cells (IDOMAIN $>$ 0) must have a K value greater than
zero.""",
    )

    k22 = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""hydraulic conductivity of second ellipsoid axis""",
        description =
"""is the hydraulic conductivity of the second ellipsoid axis (or the
ratio of K22/K if the K22OVERK option is specified); for an unrotated
case this is the hydraulic conductivity in the y direction.  If K22 is
not included in the GRIDDATA block, then K22 is set equal to K.  For a
regular MODFLOW grid (DIS Package is used) in which no rotation angles
are specified, K22 is the hydraulic conductivity along columns in the
y direction. For an unstructured DISU grid, the user must assign
principal x and y axes and provide the angle for each cell face
relative to the assigned x direction.  All included cells (IDOMAIN $>$
0) must have a K22 value greater than zero.""",
    )

    k33 = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""hydraulic conductivity of third ellipsoid axis (L/T)""",
        description =
"""is the hydraulic conductivity of the third ellipsoid axis (or the
ratio of K33/K if the K33OVERK option is specified); for an unrotated
case, this is the vertical hydraulic conductivity.  When anisotropy is
applied, K33 corresponds to the K33 tensor component.  All included
cells (IDOMAIN $>$ 0) must have a K33 value greater than zero.""",
    )

    angle1 = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""first anisotropy rotation angle (degrees)""",
        description =
"""is a rotation angle of the hydraulic conductivity tensor in degrees.
The angle represents the first of three sequential rotations of the
hydraulic conductivity ellipsoid. With the K11, K22, and K33 axes of
the ellipsoid initially aligned with the x, y, and z coordinate axes,
respectively, ANGLE1 rotates the ellipsoid about its K33 axis (within
the x - y plane). A positive value represents counter-clockwise
rotation when viewed from any point on the positive K33 axis, looking
toward the center of the ellipsoid. A value of zero indicates that the
K11 axis lies within the x - z plane. If ANGLE1 is not specified,
default values of zero are assigned to ANGLE1, ANGLE2, and ANGLE3, in
which case the K11, K22, and K33 axes are aligned with the x, y, and z
axes, respectively.""",
    )

    angle2 = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""second anisotropy rotation angle (degrees)""",
        description =
"""is a rotation angle of the hydraulic conductivity tensor in degrees.
The angle represents the second of three sequential rotations of the
hydraulic conductivity ellipsoid. Following the rotation by ANGLE1
described above, ANGLE2 rotates the ellipsoid about its K22 axis (out
of the x - y plane). An array can be specified for ANGLE2 only if
ANGLE1 is also specified. A positive value of ANGLE2 represents
clockwise rotation when viewed from any point on the positive K22
axis, looking toward the center of the ellipsoid. A value of zero
indicates that the K11 axis lies within the x - y plane. If ANGLE2 is
not specified, default values of zero are assigned to ANGLE2 and
ANGLE3; connections that are not user-designated as vertical are
assumed to be strictly horizontal (that is, to have no z component to
their orientation); and connection lengths are based on horizontal
distances.""",
    )

    angle3 = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""third anisotropy rotation angle (degrees)""",
        description =
"""is a rotation angle of the hydraulic conductivity tensor in degrees.
The angle represents the third of three sequential rotations of the
hydraulic conductivity ellipsoid. Following the rotations by ANGLE1
and ANGLE2 described above, ANGLE3 rotates the ellipsoid about its K11
axis. An array can be specified for ANGLE3 only if ANGLE1 and ANGLE2
are also specified. An array must be specified for ANGLE3 if ANGLE2 is
specified. A positive value of ANGLE3 represents clockwise rotation
when viewed from any point on the positive K11 axis, looking toward
the center of the ellipsoid. A value of zero indicates that the K22
axis lies within the x - y plane.""",
    )

    wetdry = MFArray(
        type = "double",
        block = "griddata",
        shape = "(nodes)",
        reader = "readarray",
        optional = True,
        longname =
"""wetdry threshold and factor""",
        description =
"""is a combination of the wetting threshold and a flag to indicate which
neighboring cells can cause a cell to become wet. If WETDRY $<$ 0,
only a cell below a dry cell can cause the cell to become wet. If
WETDRY $>$ 0, the cell below a dry cell and horizontally adjacent
cells can cause a cell to become wet. If WETDRY is 0, the cell cannot
be wetted. The absolute value of WETDRY is the wetting threshold. When
the sum of BOT and the absolute value of WETDRY at a dry cell is
equaled or exceeded by the head at an adjacent cell, the cell is
wetted.  WETDRY must be specified if ``REWET'' is specified in the
OPTIONS block.  If ``REWET'' is not specified in the options block,
then WETDRY can be entered, and memory will be allocated for it, even
though it is not used.""",
    )