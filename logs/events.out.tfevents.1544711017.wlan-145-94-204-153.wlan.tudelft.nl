       �K"	  @ښ�Abrain.Event:2m��C�     ]�	��Rښ�A"��
l
	natural/sPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
s
natural/Q_targetPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
�
6natural/eval_net/l1/w1/Initializer/random_normal/shapeConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
5natural/eval_net/l1/w1/Initializer/random_normal/meanConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7natural/eval_net/l1/w1/Initializer/random_normal/stddevConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Enatural/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6natural/eval_net/l1/w1/Initializer/random_normal/shape*)
_class
loc:@natural/eval_net/l1/w1*
seed2*
dtype0*
_output_shapes

:*

seed*
T0
�
4natural/eval_net/l1/w1/Initializer/random_normal/mulMulEnatural/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal7natural/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
0natural/eval_net/l1/w1/Initializer/random_normalAdd4natural/eval_net/l1/w1/Initializer/random_normal/mul5natural/eval_net/l1/w1/Initializer/random_normal/mean*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:*
T0
�
natural/eval_net/l1/w1
VariableV2*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/eval_net/l1/w1/AssignAssignnatural/eval_net/l1/w10natural/eval_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/w1*
validate_shape(*
_output_shapes

:
�
natural/eval_net/l1/w1/readIdentitynatural/eval_net/l1/w1*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
(natural/eval_net/l1/b1/Initializer/ConstConst*)
_class
loc:@natural/eval_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
natural/eval_net/l1/b1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@natural/eval_net/l1/b1*
	container 
�
natural/eval_net/l1/b1/AssignAssignnatural/eval_net/l1/b1(natural/eval_net/l1/b1/Initializer/Const*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(
�
natural/eval_net/l1/b1/readIdentitynatural/eval_net/l1/b1*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/b1
�
natural/eval_net/l1/MatMulMatMul	natural/snatural/eval_net/l1/w1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
natural/eval_net/l1/addAddnatural/eval_net/l1/MatMulnatural/eval_net/l1/b1/read*
T0*'
_output_shapes
:���������
k
natural/eval_net/l1/ReluRelunatural/eval_net/l1/add*
T0*'
_output_shapes
:���������
�
5natural/eval_net/Q/w2/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*(
_class
loc:@natural/eval_net/Q/w2*
valueB"      
�
4natural/eval_net/Q/w2/Initializer/random_normal/meanConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6natural/eval_net/Q/w2/Initializer/random_normal/stddevConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Dnatural/eval_net/Q/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5natural/eval_net/Q/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*(
_class
loc:@natural/eval_net/Q/w2*
seed2
�
3natural/eval_net/Q/w2/Initializer/random_normal/mulMulDnatural/eval_net/Q/w2/Initializer/random_normal/RandomStandardNormal6natural/eval_net/Q/w2/Initializer/random_normal/stddev*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:*
T0
�
/natural/eval_net/Q/w2/Initializer/random_normalAdd3natural/eval_net/Q/w2/Initializer/random_normal/mul4natural/eval_net/Q/w2/Initializer/random_normal/mean*
T0*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:
�
natural/eval_net/Q/w2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/w2*
	container 
�
natural/eval_net/Q/w2/AssignAssignnatural/eval_net/Q/w2/natural/eval_net/Q/w2/Initializer/random_normal*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(*
_output_shapes

:
�
natural/eval_net/Q/w2/readIdentitynatural/eval_net/Q/w2*
T0*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:
�
'natural/eval_net/Q/b2/Initializer/ConstConst*(
_class
loc:@natural/eval_net/Q/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
natural/eval_net/Q/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:
�
natural/eval_net/Q/b2/AssignAssignnatural/eval_net/Q/b2'natural/eval_net/Q/b2/Initializer/Const*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:
�
natural/eval_net/Q/b2/readIdentitynatural/eval_net/Q/b2*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:*
T0
�
natural/eval_net/Q/MatMulMatMulnatural/eval_net/l1/Relunatural/eval_net/Q/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
natural/eval_net/Q/addAddnatural/eval_net/Q/MatMulnatural/eval_net/Q/b2/read*
T0*'
_output_shapes
:���������
�
natural/loss/SquaredDifferenceSquaredDifferencenatural/Q_targetnatural/eval_net/Q/add*
T0*'
_output_shapes
:���������
c
natural/loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
natural/loss/MeanMeannatural/loss/SquaredDifferencenatural/loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
`
natural/train/gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
f
!natural/train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
natural/train/gradients/FillFillnatural/train/gradients/Shape!natural/train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
<natural/train/gradients/natural/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
6natural/train/gradients/natural/loss/Mean_grad/ReshapeReshapenatural/train/gradients/Fill<natural/train/gradients/natural/loss/Mean_grad/Reshape/shape*
Tshape0*
_output_shapes

:*
T0
�
4natural/train/gradients/natural/loss/Mean_grad/ShapeShapenatural/loss/SquaredDifference*
out_type0*
_output_shapes
:*
T0
�
3natural/train/gradients/natural/loss/Mean_grad/TileTile6natural/train/gradients/natural/loss/Mean_grad/Reshape4natural/train/gradients/natural/loss/Mean_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
�
6natural/train/gradients/natural/loss/Mean_grad/Shape_1Shapenatural/loss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
y
6natural/train/gradients/natural/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
~
4natural/train/gradients/natural/loss/Mean_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
�
3natural/train/gradients/natural/loss/Mean_grad/ProdProd6natural/train/gradients/natural/loss/Mean_grad/Shape_14natural/train/gradients/natural/loss/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
6natural/train/gradients/natural/loss/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
�
5natural/train/gradients/natural/loss/Mean_grad/Prod_1Prod6natural/train/gradients/natural/loss/Mean_grad/Shape_26natural/train/gradients/natural/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
z
8natural/train/gradients/natural/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6natural/train/gradients/natural/loss/Mean_grad/MaximumMaximum5natural/train/gradients/natural/loss/Mean_grad/Prod_18natural/train/gradients/natural/loss/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
7natural/train/gradients/natural/loss/Mean_grad/floordivFloorDiv3natural/train/gradients/natural/loss/Mean_grad/Prod6natural/train/gradients/natural/loss/Mean_grad/Maximum*
_output_shapes
: *
T0
�
3natural/train/gradients/natural/loss/Mean_grad/CastCast7natural/train/gradients/natural/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
6natural/train/gradients/natural/loss/Mean_grad/truedivRealDiv3natural/train/gradients/natural/loss/Mean_grad/Tile3natural/train/gradients/natural/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/ShapeShapenatural/Q_target*
T0*
out_type0*
_output_shapes
:
�
Cnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1Shapenatural/eval_net/Q/add*
T0*
out_type0*
_output_shapes
:
�
Qnatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsAnatural/train/gradients/natural/loss/SquaredDifference_grad/ShapeCnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bnatural/train/gradients/natural/loss/SquaredDifference_grad/scalarConst7^natural/train/gradients/natural/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/mulMulBnatural/train/gradients/natural/loss/SquaredDifference_grad/scalar6natural/train/gradients/natural/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/subSubnatural/Q_targetnatural/eval_net/Q/add7^natural/train/gradients/natural/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Mul?natural/train/gradients/natural/loss/SquaredDifference_grad/mul?natural/train/gradients/natural/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/SumSumAnatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Qnatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Cnatural/train/gradients/natural/loss/SquaredDifference_grad/ReshapeReshape?natural/train/gradients/natural/loss/SquaredDifference_grad/SumAnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/Sum_1SumAnatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Snatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Enatural/train/gradients/natural/loss/SquaredDifference_grad/Reshape_1ReshapeAnatural/train/gradients/natural/loss/SquaredDifference_grad/Sum_1Cnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/NegNegEnatural/train/gradients/natural/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Lnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_depsNoOp@^natural/train/gradients/natural/loss/SquaredDifference_grad/NegD^natural/train/gradients/natural/loss/SquaredDifference_grad/Reshape
�
Tnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependencyIdentityCnatural/train/gradients/natural/loss/SquaredDifference_grad/ReshapeM^natural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_deps*
T0*V
_classL
JHloc:@natural/train/gradients/natural/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Vnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Identity?natural/train/gradients/natural/loss/SquaredDifference_grad/NegM^natural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_deps*
T0*R
_classH
FDloc:@natural/train/gradients/natural/loss/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
9natural/train/gradients/natural/eval_net/Q/add_grad/ShapeShapenatural/eval_net/Q/MatMul*
_output_shapes
:*
T0*
out_type0
�
;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Inatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgsBroadcastGradientArgs9natural/train/gradients/natural/eval_net/Q/add_grad/Shape;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
7natural/train/gradients/natural/eval_net/Q/add_grad/SumSumVnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Inatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;natural/train/gradients/natural/eval_net/Q/add_grad/ReshapeReshape7natural/train/gradients/natural/eval_net/Q/add_grad/Sum9natural/train/gradients/natural/eval_net/Q/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9natural/train/gradients/natural/eval_net/Q/add_grad/Sum_1SumVnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Knatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1Reshape9natural/train/gradients/natural/eval_net/Q/add_grad/Sum_1;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Dnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_depsNoOp<^natural/train/gradients/natural/eval_net/Q/add_grad/Reshape>^natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1
�
Lnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependencyIdentity;natural/train/gradients/natural/eval_net/Q/add_grad/ReshapeE^natural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@natural/train/gradients/natural/eval_net/Q/add_grad/Reshape
�
Nnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency_1Identity=natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1E^natural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_deps*
_output_shapes

:*
T0*P
_classF
DBloc:@natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1
�
=natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMulMatMulLnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependencynatural/eval_net/Q/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
?natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1MatMulnatural/eval_net/l1/ReluLnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
Gnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_depsNoOp>^natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul@^natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1
�
Onatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependencyIdentity=natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMulH^natural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Qnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependency_1Identity?natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1H^natural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_deps*R
_classH
FDloc:@natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradReluGradOnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependencynatural/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
:natural/train/gradients/natural/eval_net/l1/add_grad/ShapeShapenatural/eval_net/l1/MatMul*
_output_shapes
:*
T0*
out_type0
�
<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Jnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:natural/train/gradients/natural/eval_net/l1/add_grad/Shape<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8natural/train/gradients/natural/eval_net/l1/add_grad/SumSum>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradJnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<natural/train/gradients/natural/eval_net/l1/add_grad/ReshapeReshape8natural/train/gradients/natural/eval_net/l1/add_grad/Sum:natural/train/gradients/natural/eval_net/l1/add_grad/Shape*
Tshape0*'
_output_shapes
:���������*
T0
�
:natural/train/gradients/natural/eval_net/l1/add_grad/Sum_1Sum>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradLnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1Reshape:natural/train/gradients/natural/eval_net/l1/add_grad/Sum_1<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Enatural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_depsNoOp=^natural/train/gradients/natural/eval_net/l1/add_grad/Reshape?^natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1
�
Mnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependencyIdentity<natural/train/gradients/natural/eval_net/l1/add_grad/ReshapeF^natural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_deps*O
_classE
CAloc:@natural/train/gradients/natural/eval_net/l1/add_grad/Reshape*'
_output_shapes
:���������*
T0
�
Onatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency_1Identity>natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1F^natural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:
�
>natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulMatMulMnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependencynatural/eval_net/l1/w1/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1MatMul	natural/sMnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
Hnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_depsNoOp?^natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulA^natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1
�
Pnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity>natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulI^natural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*Q
_classG
ECloc:@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul
�
Rnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1I^natural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:
�
=natural/train/natural/eval_net/l1/w1/RMSProp/Initializer/onesConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB*  �?*
dtype0*
_output_shapes

:
�
,natural/train/natural/eval_net/l1/w1/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container *
shape
:
�
3natural/train/natural/eval_net/l1/w1/RMSProp/AssignAssign,natural/train/natural/eval_net/l1/w1/RMSProp=natural/train/natural/eval_net/l1/w1/RMSProp/Initializer/ones*
_output_shapes

:*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/w1*
validate_shape(
�
1natural/train/natural/eval_net/l1/w1/RMSProp/readIdentity,natural/train/natural/eval_net/l1/w1/RMSProp*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/w1
�
@natural/train/natural/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB*    *
dtype0*
_output_shapes

:
�
.natural/train/natural/eval_net/l1/w1/RMSProp_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container 
�
5natural/train/natural/eval_net/l1/w1/RMSProp_1/AssignAssign.natural/train/natural/eval_net/l1/w1/RMSProp_1@natural/train/natural/eval_net/l1/w1/RMSProp_1/Initializer/zeros*)
_class
loc:@natural/eval_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
3natural/train/natural/eval_net/l1/w1/RMSProp_1/readIdentity.natural/train/natural/eval_net/l1/w1/RMSProp_1*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
=natural/train/natural/eval_net/l1/b1/RMSProp/Initializer/onesConst*)
_class
loc:@natural/eval_net/l1/b1*
valueB*  �?*
dtype0*
_output_shapes

:
�
,natural/train/natural/eval_net/l1/b1/RMSProp
VariableV2*
shared_name *)
_class
loc:@natural/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
3natural/train/natural/eval_net/l1/b1/RMSProp/AssignAssign,natural/train/natural/eval_net/l1/b1/RMSProp=natural/train/natural/eval_net/l1/b1/RMSProp/Initializer/ones*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
1natural/train/natural/eval_net/l1/b1/RMSProp/readIdentity,natural/train/natural/eval_net/l1/b1/RMSProp*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/b1
�
@natural/train/natural/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@natural/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
.natural/train/natural/eval_net/l1/b1/RMSProp_1
VariableV2*)
_class
loc:@natural/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
5natural/train/natural/eval_net/l1/b1/RMSProp_1/AssignAssign.natural/train/natural/eval_net/l1/b1/RMSProp_1@natural/train/natural/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
3natural/train/natural/eval_net/l1/b1/RMSProp_1/readIdentity.natural/train/natural/eval_net/l1/b1/RMSProp_1*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/b1
�
<natural/train/natural/eval_net/Q/w2/RMSProp/Initializer/onesConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB*  �?*
dtype0*
_output_shapes

:
�
+natural/train/natural/eval_net/Q/w2/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/w2*
	container *
shape
:
�
2natural/train/natural/eval_net/Q/w2/RMSProp/AssignAssign+natural/train/natural/eval_net/Q/w2/RMSProp<natural/train/natural/eval_net/Q/w2/RMSProp/Initializer/ones*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(*
_output_shapes

:*
use_locking(
�
0natural/train/natural/eval_net/Q/w2/RMSProp/readIdentity+natural/train/natural/eval_net/Q/w2/RMSProp*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:*
T0
�
?natural/train/natural/eval_net/Q/w2/RMSProp_1/Initializer/zerosConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB*    *
dtype0*
_output_shapes

:
�
-natural/train/natural/eval_net/Q/w2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/w2*
	container *
shape
:
�
4natural/train/natural/eval_net/Q/w2/RMSProp_1/AssignAssign-natural/train/natural/eval_net/Q/w2/RMSProp_1?natural/train/natural/eval_net/Q/w2/RMSProp_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(
�
2natural/train/natural/eval_net/Q/w2/RMSProp_1/readIdentity-natural/train/natural/eval_net/Q/w2/RMSProp_1*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:*
T0
�
<natural/train/natural/eval_net/Q/b2/RMSProp/Initializer/onesConst*(
_class
loc:@natural/eval_net/Q/b2*
valueB*  �?*
dtype0*
_output_shapes

:
�
+natural/train/natural/eval_net/Q/b2/RMSProp
VariableV2*(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
2natural/train/natural/eval_net/Q/b2/RMSProp/AssignAssign+natural/train/natural/eval_net/Q/b2/RMSProp<natural/train/natural/eval_net/Q/b2/RMSProp/Initializer/ones*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:*
use_locking(
�
0natural/train/natural/eval_net/Q/b2/RMSProp/readIdentity+natural/train/natural/eval_net/Q/b2/RMSProp*
T0*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:
�
?natural/train/natural/eval_net/Q/b2/RMSProp_1/Initializer/zerosConst*
_output_shapes

:*(
_class
loc:@natural/eval_net/Q/b2*
valueB*    *
dtype0
�
-natural/train/natural/eval_net/Q/b2/RMSProp_1
VariableV2*
shared_name *(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
4natural/train/natural/eval_net/Q/b2/RMSProp_1/AssignAssign-natural/train/natural/eval_net/Q/b2/RMSProp_1?natural/train/natural/eval_net/Q/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:
�
2natural/train/natural/eval_net/Q/b2/RMSProp_1/readIdentity-natural/train/natural/eval_net/Q/b2/RMSProp_1*
T0*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:
h
#natural/train/RMSProp/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
`
natural/train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
c
natural/train/RMSProp/momentumConst*
dtype0*
_output_shapes
: *
valueB
 *    
b
natural/train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
@natural/train/RMSProp/update_natural/eval_net/l1/w1/ApplyRMSPropApplyRMSPropnatural/eval_net/l1/w1,natural/train/natural/eval_net/l1/w1/RMSProp.natural/train/natural/eval_net/l1/w1/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonRnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
@natural/train/RMSProp/update_natural/eval_net/l1/b1/ApplyRMSPropApplyRMSPropnatural/eval_net/l1/b1,natural/train/natural/eval_net/l1/b1/RMSProp.natural/train/natural/eval_net/l1/b1/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonOnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@natural/eval_net/l1/b1*
_output_shapes

:
�
?natural/train/RMSProp/update_natural/eval_net/Q/w2/ApplyRMSPropApplyRMSPropnatural/eval_net/Q/w2+natural/train/natural/eval_net/Q/w2/RMSProp-natural/train/natural/eval_net/Q/w2/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonQnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*(
_class
loc:@natural/eval_net/Q/w2
�
?natural/train/RMSProp/update_natural/eval_net/Q/b2/ApplyRMSPropApplyRMSPropnatural/eval_net/Q/b2+natural/train/natural/eval_net/Q/b2/RMSProp-natural/train/natural/eval_net/Q/b2/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonNnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:
�
natural/train/RMSPropNoOp@^natural/train/RMSProp/update_natural/eval_net/Q/b2/ApplyRMSProp@^natural/train/RMSProp/update_natural/eval_net/Q/w2/ApplyRMSPropA^natural/train/RMSProp/update_natural/eval_net/l1/b1/ApplyRMSPropA^natural/train/RMSProp/update_natural/eval_net/l1/w1/ApplyRMSProp
m

natural/s_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
8natural/target_net/l1/w1/Initializer/random_normal/shapeConst*+
_class!
loc:@natural/target_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
7natural/target_net/l1/w1/Initializer/random_normal/meanConst*
_output_shapes
: *+
_class!
loc:@natural/target_net/l1/w1*
valueB
 *    *
dtype0
�
9natural/target_net/l1/w1/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *+
_class!
loc:@natural/target_net/l1/w1*
valueB
 *���>
�
Gnatural/target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8natural/target_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*+
_class!
loc:@natural/target_net/l1/w1*
seed2�
�
6natural/target_net/l1/w1/Initializer/random_normal/mulMulGnatural/target_net/l1/w1/Initializer/random_normal/RandomStandardNormal9natural/target_net/l1/w1/Initializer/random_normal/stddev*
T0*+
_class!
loc:@natural/target_net/l1/w1*
_output_shapes

:
�
2natural/target_net/l1/w1/Initializer/random_normalAdd6natural/target_net/l1/w1/Initializer/random_normal/mul7natural/target_net/l1/w1/Initializer/random_normal/mean*+
_class!
loc:@natural/target_net/l1/w1*
_output_shapes

:*
T0
�
natural/target_net/l1/w1
VariableV2*
shared_name *+
_class!
loc:@natural/target_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/target_net/l1/w1/AssignAssignnatural/target_net/l1/w12natural/target_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:
�
natural/target_net/l1/w1/readIdentitynatural/target_net/l1/w1*
_output_shapes

:*
T0*+
_class!
loc:@natural/target_net/l1/w1
�
*natural/target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:*+
_class!
loc:@natural/target_net/l1/b1*
valueB*���=
�
natural/target_net/l1/b1
VariableV2*
shared_name *+
_class!
loc:@natural/target_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/target_net/l1/b1/AssignAssignnatural/target_net/l1/b1*natural/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
natural/target_net/l1/b1/readIdentitynatural/target_net/l1/b1*
T0*+
_class!
loc:@natural/target_net/l1/b1*
_output_shapes

:
�
natural/target_net/l1/MatMulMatMul
natural/s_natural/target_net/l1/w1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
natural/target_net/l1/addAddnatural/target_net/l1/MatMulnatural/target_net/l1/b1/read*
T0*'
_output_shapes
:���������
o
natural/target_net/l1/ReluRelunatural/target_net/l1/add*
T0*'
_output_shapes
:���������
�
7natural/target_net/Q/w2/Initializer/random_normal/shapeConst**
_class 
loc:@natural/target_net/Q/w2*
valueB"      *
dtype0*
_output_shapes
:
�
6natural/target_net/Q/w2/Initializer/random_normal/meanConst**
_class 
loc:@natural/target_net/Q/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8natural/target_net/Q/w2/Initializer/random_normal/stddevConst**
_class 
loc:@natural/target_net/Q/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Fnatural/target_net/Q/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7natural/target_net/Q/w2/Initializer/random_normal/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0**
_class 
loc:@natural/target_net/Q/w2
�
5natural/target_net/Q/w2/Initializer/random_normal/mulMulFnatural/target_net/Q/w2/Initializer/random_normal/RandomStandardNormal8natural/target_net/Q/w2/Initializer/random_normal/stddev*
T0**
_class 
loc:@natural/target_net/Q/w2*
_output_shapes

:
�
1natural/target_net/Q/w2/Initializer/random_normalAdd5natural/target_net/Q/w2/Initializer/random_normal/mul6natural/target_net/Q/w2/Initializer/random_normal/mean*
T0**
_class 
loc:@natural/target_net/Q/w2*
_output_shapes

:
�
natural/target_net/Q/w2
VariableV2*
shared_name **
_class 
loc:@natural/target_net/Q/w2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/target_net/Q/w2/AssignAssignnatural/target_net/Q/w21natural/target_net/Q/w2/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/w2
�
natural/target_net/Q/w2/readIdentitynatural/target_net/Q/w2**
_class 
loc:@natural/target_net/Q/w2*
_output_shapes

:*
T0
�
)natural/target_net/Q/b2/Initializer/ConstConst**
_class 
loc:@natural/target_net/Q/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
natural/target_net/Q/b2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@natural/target_net/Q/b2*
	container 
�
natural/target_net/Q/b2/AssignAssignnatural/target_net/Q/b2)natural/target_net/Q/b2/Initializer/Const**
_class 
loc:@natural/target_net/Q/b2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
natural/target_net/Q/b2/readIdentitynatural/target_net/Q/b2*
T0**
_class 
loc:@natural/target_net/Q/b2*
_output_shapes

:
�
natural/target_net/Q/MatMulMatMulnatural/target_net/l1/Relunatural/target_net/Q/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
natural/target_net/Q/addAddnatural/target_net/Q/MatMulnatural/target_net/Q/b2/read*
T0*'
_output_shapes
:���������
�
natural/AssignAssignnatural/target_net/l1/w1natural/eval_net/l1/w1/read*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
natural/Assign_1Assignnatural/target_net/l1/b1natural/eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/b1
�
natural/Assign_2Assignnatural/target_net/Q/w2natural/eval_net/Q/w2/read*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/w2*
validate_shape(*
_output_shapes

:
�
natural/Assign_3Assignnatural/target_net/Q/b2natural/eval_net/Q/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/b2
l
	dueling/sPlaceholder*'
_output_shapes
:���������*
shape:���������*
dtype0
s
dueling/Q_targetPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
6dueling/eval_net/l1/w1/Initializer/random_normal/shapeConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
5dueling/eval_net/l1/w1/Initializer/random_normal/meanConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7dueling/eval_net/l1/w1/Initializer/random_normal/stddevConst*
_output_shapes
: *)
_class
loc:@dueling/eval_net/l1/w1*
valueB
 *���>*
dtype0
�
Edueling/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6dueling/eval_net/l1/w1/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
seed2�*
dtype0
�
4dueling/eval_net/l1/w1/Initializer/random_normal/mulMulEdueling/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal7dueling/eval_net/l1/w1/Initializer/random_normal/stddev*
_output_shapes

:*
T0*)
_class
loc:@dueling/eval_net/l1/w1
�
0dueling/eval_net/l1/w1/Initializer/random_normalAdd4dueling/eval_net/l1/w1/Initializer/random_normal/mul5dueling/eval_net/l1/w1/Initializer/random_normal/mean*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
dueling/eval_net/l1/w1
VariableV2*)
_class
loc:@dueling/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
dueling/eval_net/l1/w1/AssignAssigndueling/eval_net/l1/w10dueling/eval_net/l1/w1/Initializer/random_normal*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/eval_net/l1/w1/readIdentitydueling/eval_net/l1/w1*
_output_shapes

:*
T0*)
_class
loc:@dueling/eval_net/l1/w1
�
(dueling/eval_net/l1/b1/Initializer/ConstConst*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/eval_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1
�
dueling/eval_net/l1/b1/AssignAssigndueling/eval_net/l1/b1(dueling/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
dueling/eval_net/l1/b1/readIdentitydueling/eval_net/l1/b1*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
dueling/eval_net/l1/MatMulMatMul	dueling/sdueling/eval_net/l1/w1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/eval_net/l1/addAdddueling/eval_net/l1/MatMuldueling/eval_net/l1/b1/read*
T0*'
_output_shapes
:���������
k
dueling/eval_net/l1/ReluReludueling/eval_net/l1/add*
T0*'
_output_shapes
:���������
�
9dueling/eval_net/Value/w2/Initializer/random_normal/shapeConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB"      *
dtype0*
_output_shapes
:
�
8dueling/eval_net/Value/w2/Initializer/random_normal/meanConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
:dueling/eval_net/Value/w2/Initializer/random_normal/stddevConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Hdueling/eval_net/Value/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9dueling/eval_net/Value/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
seed2�
�
7dueling/eval_net/Value/w2/Initializer/random_normal/mulMulHdueling/eval_net/Value/w2/Initializer/random_normal/RandomStandardNormal:dueling/eval_net/Value/w2/Initializer/random_normal/stddev*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
3dueling/eval_net/Value/w2/Initializer/random_normalAdd7dueling/eval_net/Value/w2/Initializer/random_normal/mul8dueling/eval_net/Value/w2/Initializer/random_normal/mean*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
dueling/eval_net/Value/w2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/w2*
	container 
�
 dueling/eval_net/Value/w2/AssignAssigndueling/eval_net/Value/w23dueling/eval_net/Value/w2/Initializer/random_normal*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:
�
dueling/eval_net/Value/w2/readIdentitydueling/eval_net/Value/w2*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
+dueling/eval_net/Value/b2/Initializer/ConstConst*
_output_shapes

:*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*���=*
dtype0
�
dueling/eval_net/Value/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/b2*
	container *
shape
:
�
 dueling/eval_net/Value/b2/AssignAssigndueling/eval_net/Value/b2+dueling/eval_net/Value/b2/Initializer/Const*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(*
_output_shapes

:
�
dueling/eval_net/Value/b2/readIdentitydueling/eval_net/Value/b2*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
dueling/eval_net/Value/MatMulMatMuldueling/eval_net/l1/Reludueling/eval_net/Value/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/eval_net/Value/addAdddueling/eval_net/Value/MatMuldueling/eval_net/Value/b2/read*
T0*'
_output_shapes
:���������
�
=dueling/eval_net/Advantage/w2/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB"      
�
<dueling/eval_net/Advantage/w2/Initializer/random_normal/meanConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
>dueling/eval_net/Advantage/w2/Initializer/random_normal/stddevConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Ldueling/eval_net/Advantage/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=dueling/eval_net/Advantage/w2/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
seed2�*
dtype0
�
;dueling/eval_net/Advantage/w2/Initializer/random_normal/mulMulLdueling/eval_net/Advantage/w2/Initializer/random_normal/RandomStandardNormal>dueling/eval_net/Advantage/w2/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
7dueling/eval_net/Advantage/w2/Initializer/random_normalAdd;dueling/eval_net/Advantage/w2/Initializer/random_normal/mul<dueling/eval_net/Advantage/w2/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
dueling/eval_net/Advantage/w2
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/w2
�
$dueling/eval_net/Advantage/w2/AssignAssigndueling/eval_net/Advantage/w27dueling/eval_net/Advantage/w2/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
"dueling/eval_net/Advantage/w2/readIdentitydueling/eval_net/Advantage/w2*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:*
T0
�
/dueling/eval_net/Advantage/b2/Initializer/ConstConst*
_output_shapes

:*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*���=*
dtype0
�
dueling/eval_net/Advantage/b2
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2
�
$dueling/eval_net/Advantage/b2/AssignAssigndueling/eval_net/Advantage/b2/dueling/eval_net/Advantage/b2/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
"dueling/eval_net/Advantage/b2/readIdentitydueling/eval_net/Advantage/b2*
_output_shapes

:*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2
�
!dueling/eval_net/Advantage/MatMulMatMuldueling/eval_net/l1/Relu"dueling/eval_net/Advantage/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/eval_net/Advantage/addAdd!dueling/eval_net/Advantage/MatMul"dueling/eval_net/Advantage/b2/read*
T0*'
_output_shapes
:���������
k
)dueling/eval_net/Q/Mean/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dueling/eval_net/Q/MeanMeandueling/eval_net/Advantage/add)dueling/eval_net/Q/Mean/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
dueling/eval_net/Q/subSubdueling/eval_net/Advantage/adddueling/eval_net/Q/Mean*'
_output_shapes
:���������*
T0
�
dueling/eval_net/Q/addAdddueling/eval_net/Value/adddueling/eval_net/Q/sub*'
_output_shapes
:���������*
T0
�
dueling/loss/SquaredDifferenceSquaredDifferencedueling/Q_targetdueling/eval_net/Q/add*'
_output_shapes
:���������*
T0
c
dueling/loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dueling/loss/MeanMeandueling/loss/SquaredDifferencedueling/loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
`
dueling/train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
f
!dueling/train/gradients/grad_ys_0Const*
_output_shapes
: *
valueB
 *  �?*
dtype0
�
dueling/train/gradients/FillFilldueling/train/gradients/Shape!dueling/train/gradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
�
<dueling/train/gradients/dueling/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
6dueling/train/gradients/dueling/loss/Mean_grad/ReshapeReshapedueling/train/gradients/Fill<dueling/train/gradients/dueling/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
4dueling/train/gradients/dueling/loss/Mean_grad/ShapeShapedueling/loss/SquaredDifference*
_output_shapes
:*
T0*
out_type0
�
3dueling/train/gradients/dueling/loss/Mean_grad/TileTile6dueling/train/gradients/dueling/loss/Mean_grad/Reshape4dueling/train/gradients/dueling/loss/Mean_grad/Shape*'
_output_shapes
:���������*

Tmultiples0*
T0
�
6dueling/train/gradients/dueling/loss/Mean_grad/Shape_1Shapedueling/loss/SquaredDifference*
_output_shapes
:*
T0*
out_type0
y
6dueling/train/gradients/dueling/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
~
4dueling/train/gradients/dueling/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
3dueling/train/gradients/dueling/loss/Mean_grad/ProdProd6dueling/train/gradients/dueling/loss/Mean_grad/Shape_14dueling/train/gradients/dueling/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
6dueling/train/gradients/dueling/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
5dueling/train/gradients/dueling/loss/Mean_grad/Prod_1Prod6dueling/train/gradients/dueling/loss/Mean_grad/Shape_26dueling/train/gradients/dueling/loss/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
z
8dueling/train/gradients/dueling/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6dueling/train/gradients/dueling/loss/Mean_grad/MaximumMaximum5dueling/train/gradients/dueling/loss/Mean_grad/Prod_18dueling/train/gradients/dueling/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
7dueling/train/gradients/dueling/loss/Mean_grad/floordivFloorDiv3dueling/train/gradients/dueling/loss/Mean_grad/Prod6dueling/train/gradients/dueling/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
3dueling/train/gradients/dueling/loss/Mean_grad/CastCast7dueling/train/gradients/dueling/loss/Mean_grad/floordiv*
_output_shapes
: *

DstT0*

SrcT0
�
6dueling/train/gradients/dueling/loss/Mean_grad/truedivRealDiv3dueling/train/gradients/dueling/loss/Mean_grad/Tile3dueling/train/gradients/dueling/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/ShapeShapedueling/Q_target*
_output_shapes
:*
T0*
out_type0
�
Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1Shapedueling/eval_net/Q/add*
T0*
out_type0*
_output_shapes
:
�
Qdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsAdueling/train/gradients/dueling/loss/SquaredDifference_grad/ShapeCdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
Bdueling/train/gradients/dueling/loss/SquaredDifference_grad/scalarConst7^dueling/train/gradients/dueling/loss/Mean_grad/truediv*
dtype0*
_output_shapes
: *
valueB
 *   @
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/mulMulBdueling/train/gradients/dueling/loss/SquaredDifference_grad/scalar6dueling/train/gradients/dueling/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/subSubdueling/Q_targetdueling/eval_net/Q/add7^dueling/train/gradients/dueling/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Mul?dueling/train/gradients/dueling/loss/SquaredDifference_grad/mul?dueling/train/gradients/dueling/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/SumSumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Qdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/ReshapeReshape?dueling/train/gradients/dueling/loss/SquaredDifference_grad/SumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/Sum_1SumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Sdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Edueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape_1ReshapeAdueling/train/gradients/dueling/loss/SquaredDifference_grad/Sum_1Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegNegEdueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Ldueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_depsNoOp@^dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegD^dueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape
�
Tdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependencyIdentityCdueling/train/gradients/dueling/loss/SquaredDifference_grad/ReshapeM^dueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*V
_classL
JHloc:@dueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape
�
Vdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Identity?dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegM^dueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_deps*
T0*R
_classH
FDloc:@dueling/train/gradients/dueling/loss/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
9dueling/train/gradients/dueling/eval_net/Q/add_grad/ShapeShapedueling/eval_net/Value/add*
_output_shapes
:*
T0*
out_type0
�
;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1Shapedueling/eval_net/Q/sub*
out_type0*
_output_shapes
:*
T0
�
Idueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgsBroadcastGradientArgs9dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7dueling/train/gradients/dueling/eval_net/Q/add_grad/SumSumVdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Idueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;dueling/train/gradients/dueling/eval_net/Q/add_grad/ReshapeReshape7dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum9dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
9dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum_1SumVdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Kdueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
=dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1Reshape9dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum_1;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
�
Ddueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_depsNoOp<^dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape>^dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1
�
Ldueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyIdentity;dueling/train/gradients/dueling/eval_net/Q/add_grad/ReshapeE^dueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape
�
Ndueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Identity=dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1E^dueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1*'
_output_shapes
:���������
�
=dueling/train/gradients/dueling/eval_net/Value/add_grad/ShapeShapedueling/eval_net/Value/MatMul*
T0*
out_type0*
_output_shapes
:
�
?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Mdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgsBroadcastGradientArgs=dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
;dueling/train/gradients/dueling/eval_net/Value/add_grad/SumSumLdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyMdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
?dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeReshape;dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum=dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
=dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum_1SumLdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyOdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Adueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1Reshape=dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum_1?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Hdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_depsNoOp@^dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeB^dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1
�
Pdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependencyIdentity?dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeI^dueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_deps*R
_classH
FDloc:@dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape*'
_output_shapes
:���������*
T0
�
Rdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency_1IdentityAdueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1I^dueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_deps*
_output_shapes

:*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1
�
9dueling/train/gradients/dueling/eval_net/Q/sub_grad/ShapeShapedueling/eval_net/Advantage/add*
_output_shapes
:*
T0*
out_type0
�
;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1Shapedueling/eval_net/Q/Mean*
_output_shapes
:*
T0*
out_type0
�
Idueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7dueling/train/gradients/dueling/eval_net/Q/sub_grad/SumSumNdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Idueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;dueling/train/gradients/dueling/eval_net/Q/sub_grad/ReshapeReshape7dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum_1SumNdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Kdueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7dueling/train/gradients/dueling/eval_net/Q/sub_grad/NegNeg9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum_1*
T0*
_output_shapes
:
�
=dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1Reshape7dueling/train/gradients/dueling/eval_net/Q/sub_grad/Neg;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_depsNoOp<^dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape>^dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1
�
Ldueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependencyIdentity;dueling/train/gradients/dueling/eval_net/Q/sub_grad/ReshapeE^dueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_deps*
T0*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape*'
_output_shapes
:���������
�
Ndueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency_1Identity=dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1E^dueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_deps*
T0*P
_classF
DBloc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulMatMulPdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependencydueling/eval_net/Value/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
Cdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1MatMuldueling/eval_net/l1/ReluPdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
Kdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_depsNoOpB^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulD^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1
�
Sdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependencyIdentityAdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulL^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Udueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependency_1IdentityCdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1L^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1*
_output_shapes

:
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ShapeShapedueling/eval_net/Advantage/add*
_output_shapes
:*
T0*
out_type0
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/SizeConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/addAdd)dueling/eval_net/Q/Mean/reduction_indices9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
: 
�
8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/modFloorMod8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/add9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size*
_output_shapes
: *
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_1Const*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/startConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/deltaConst*
dtype0*
_output_shapes
: *M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/rangeRange@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/start9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/delta*

Tidx0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
:
�
?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill/valueConst*
dtype0*
_output_shapes
: *M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/FillFill<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_1?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill/value*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*

index_type0*
_output_shapes
: 
�
Bdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitchDynamicStitch:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/mod:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
N*#
_output_shapes
:���������*
T0
�
>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum/yConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/MaximumMaximumBdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitch>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum/y*#
_output_shapes
:���������*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape
�
=dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordivFloorDiv:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
:
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ReshapeReshapeNdueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency_1Bdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/TileTile<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Reshape=dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv*
T0*0
_output_shapes
:������������������*

Tmultiples0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_2Shapedueling/eval_net/Advantage/add*
_output_shapes
:*
T0*
out_type0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_3Shapedueling/eval_net/Q/Mean*
T0*
out_type0*
_output_shapes
:
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ProdProd<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_2:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
;dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod_1Prod<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_3<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1Maximum;dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod_1@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv_1FloorDiv9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1*
T0*
_output_shapes
: 
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/CastCast?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv_1*

SrcT0*
_output_shapes
: *

DstT0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/truedivRealDiv9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Tile9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
dueling/train/gradients/AddNAddNLdueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/truediv*
N*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape
�
Adueling/train/gradients/dueling/eval_net/Advantage/add_grad/ShapeShape!dueling/eval_net/Advantage/MatMul*
T0*
out_type0*
_output_shapes
:
�
Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Qdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgsBroadcastGradientArgsAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ShapeCdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
?dueling/train/gradients/dueling/eval_net/Advantage/add_grad/SumSumdueling/train/gradients/AddNQdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeReshape?dueling/train/gradients/dueling/eval_net/Advantage/add_grad/SumAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Adueling/train/gradients/dueling/eval_net/Advantage/add_grad/Sum_1Sumdueling/train/gradients/AddNSdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Edueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1ReshapeAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Sum_1Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1*
Tshape0*
_output_shapes

:*
T0
�
Ldueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_depsNoOpD^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeF^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1
�
Tdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependencyIdentityCdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeM^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*V
_classL
JHloc:@dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape
�
Vdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency_1IdentityEdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1M^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_deps*X
_classN
LJloc:@dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1*
_output_shapes

:*
T0
�
Edueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulMatMulTdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency"dueling/eval_net/Advantage/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
Gdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1MatMuldueling/eval_net/l1/ReluTdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
Odueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_depsNoOpF^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulH^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1
�
Wdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependencyIdentityEdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulP^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Ydueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency_1IdentityGdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1P^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1*
_output_shapes

:
�
dueling/train/gradients/AddN_1AddNSdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependencyWdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul*
N*'
_output_shapes
:���������
�
>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradReluGraddueling/train/gradients/AddN_1dueling/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
:dueling/train/gradients/dueling/eval_net/l1/add_grad/ShapeShapedueling/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
�
<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Jdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8dueling/train/gradients/dueling/eval_net/l1/add_grad/SumSum>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradJdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
<dueling/train/gradients/dueling/eval_net/l1/add_grad/ReshapeReshape8dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum:dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
:dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum_1Sum>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradLdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1Reshape:dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum_1<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Edueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_depsNoOp=^dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape?^dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1
�
Mdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependencyIdentity<dueling/train/gradients/dueling/eval_net/l1/add_grad/ReshapeF^dueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*O
_classE
CAloc:@dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape
�
Odueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency_1Identity>dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1F^dueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:
�
>dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulMatMulMdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependencydueling/eval_net/l1/w1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1MatMul	dueling/sMdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
Hdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_depsNoOp?^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulA^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1
�
Pdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity>dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulI^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Rdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1I^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:
�
=dueling/train/dueling/eval_net/l1/w1/RMSProp/Initializer/onesConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB*  �?*
dtype0*
_output_shapes

:
�
,dueling/train/dueling/eval_net/l1/w1/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/w1*
	container *
shape
:
�
3dueling/train/dueling/eval_net/l1/w1/RMSProp/AssignAssign,dueling/train/dueling/eval_net/l1/w1/RMSProp=dueling/train/dueling/eval_net/l1/w1/RMSProp/Initializer/ones*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
validate_shape(*
_output_shapes

:
�
1dueling/train/dueling/eval_net/l1/w1/RMSProp/readIdentity,dueling/train/dueling/eval_net/l1/w1/RMSProp*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:*
T0
�
@dueling/train/dueling/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB*    *
dtype0*
_output_shapes

:
�
.dueling/train/dueling/eval_net/l1/w1/RMSProp_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/w1*
	container 
�
5dueling/train/dueling/eval_net/l1/w1/RMSProp_1/AssignAssign.dueling/train/dueling/eval_net/l1/w1/RMSProp_1@dueling/train/dueling/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
3dueling/train/dueling/eval_net/l1/w1/RMSProp_1/readIdentity.dueling/train/dueling/eval_net/l1/w1/RMSProp_1*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
=dueling/train/dueling/eval_net/l1/b1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*  �?
�
,dueling/train/dueling/eval_net/l1/b1/RMSProp
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1*
	container 
�
3dueling/train/dueling/eval_net/l1/b1/RMSProp/AssignAssign,dueling/train/dueling/eval_net/l1/b1/RMSProp=dueling/train/dueling/eval_net/l1/b1/RMSProp/Initializer/ones*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(
�
1dueling/train/dueling/eval_net/l1/b1/RMSProp/readIdentity,dueling/train/dueling/eval_net/l1/b1/RMSProp*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
@dueling/train/dueling/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
.dueling/train/dueling/eval_net/l1/b1/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1*
	container *
shape
:
�
5dueling/train/dueling/eval_net/l1/b1/RMSProp_1/AssignAssign.dueling/train/dueling/eval_net/l1/b1/RMSProp_1@dueling/train/dueling/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
3dueling/train/dueling/eval_net/l1/b1/RMSProp_1/readIdentity.dueling/train/dueling/eval_net/l1/b1/RMSProp_1*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
@dueling/train/dueling/eval_net/Value/w2/RMSProp/Initializer/onesConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB*  �?*
dtype0*
_output_shapes

:
�
/dueling/train/dueling/eval_net/Value/w2/RMSProp
VariableV2*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/w2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
6dueling/train/dueling/eval_net/Value/w2/RMSProp/AssignAssign/dueling/train/dueling/eval_net/Value/w2/RMSProp@dueling/train/dueling/eval_net/Value/w2/RMSProp/Initializer/ones*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:
�
4dueling/train/dueling/eval_net/Value/w2/RMSProp/readIdentity/dueling/train/dueling/eval_net/Value/w2/RMSProp*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:*
T0
�
Cdueling/train/dueling/eval_net/Value/w2/RMSProp_1/Initializer/zerosConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB*    *
dtype0*
_output_shapes

:
�
1dueling/train/dueling/eval_net/Value/w2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/w2*
	container *
shape
:
�
8dueling/train/dueling/eval_net/Value/w2/RMSProp_1/AssignAssign1dueling/train/dueling/eval_net/Value/w2/RMSProp_1Cdueling/train/dueling/eval_net/Value/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:
�
6dueling/train/dueling/eval_net/Value/w2/RMSProp_1/readIdentity1dueling/train/dueling/eval_net/Value/w2/RMSProp_1*
_output_shapes

:*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2
�
@dueling/train/dueling/eval_net/Value/b2/RMSProp/Initializer/onesConst*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*  �?*
dtype0*
_output_shapes

:
�
/dueling/train/dueling/eval_net/Value/b2/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/b2*
	container *
shape
:
�
6dueling/train/dueling/eval_net/Value/b2/RMSProp/AssignAssign/dueling/train/dueling/eval_net/Value/b2/RMSProp@dueling/train/dueling/eval_net/Value/b2/RMSProp/Initializer/ones*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(*
_output_shapes

:*
use_locking(
�
4dueling/train/dueling/eval_net/Value/b2/RMSProp/readIdentity/dueling/train/dueling/eval_net/Value/b2/RMSProp*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
Cdueling/train/dueling/eval_net/Value/b2/RMSProp_1/Initializer/zerosConst*
_output_shapes

:*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*    *
dtype0
�
1dueling/train/dueling/eval_net/Value/b2/RMSProp_1
VariableV2*,
_class"
 loc:@dueling/eval_net/Value/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
8dueling/train/dueling/eval_net/Value/b2/RMSProp_1/AssignAssign1dueling/train/dueling/eval_net/Value/b2/RMSProp_1Cdueling/train/dueling/eval_net/Value/b2/RMSProp_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(
�
6dueling/train/dueling/eval_net/Value/b2/RMSProp_1/readIdentity1dueling/train/dueling/eval_net/Value/b2/RMSProp_1*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
Ddueling/train/dueling/eval_net/Advantage/w2/RMSProp/Initializer/onesConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB*  �?*
dtype0*
_output_shapes

:
�
3dueling/train/dueling/eval_net/Advantage/w2/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
	container *
shape
:
�
:dueling/train/dueling/eval_net/Advantage/w2/RMSProp/AssignAssign3dueling/train/dueling/eval_net/Advantage/w2/RMSPropDdueling/train/dueling/eval_net/Advantage/w2/RMSProp/Initializer/ones*
_output_shapes

:*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(
�
8dueling/train/dueling/eval_net/Advantage/w2/RMSProp/readIdentity3dueling/train/dueling/eval_net/Advantage/w2/RMSProp*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
Gdueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Initializer/zerosConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB*    *
dtype0*
_output_shapes

:
�
5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1
VariableV2*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
<dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/AssignAssign5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1Gdueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
:dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/readIdentity5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1*
_output_shapes

:*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2
�
Ddueling/train/dueling/eval_net/Advantage/b2/RMSProp/Initializer/onesConst*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*  �?*
dtype0*
_output_shapes

:
�
3dueling/train/dueling/eval_net/Advantage/b2/RMSProp
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
	container 
�
:dueling/train/dueling/eval_net/Advantage/b2/RMSProp/AssignAssign3dueling/train/dueling/eval_net/Advantage/b2/RMSPropDdueling/train/dueling/eval_net/Advantage/b2/RMSProp/Initializer/ones*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
8dueling/train/dueling/eval_net/Advantage/b2/RMSProp/readIdentity3dueling/train/dueling/eval_net/Advantage/b2/RMSProp*
_output_shapes

:*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2
�
Gdueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*    
�
5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
	container *
shape
:
�
<dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/AssignAssign5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1Gdueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Initializer/zeros*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
:dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/readIdentity5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
_output_shapes

:
h
#dueling/train/RMSProp/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
`
dueling/train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
c
dueling/train/RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
dueling/train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
@dueling/train/RMSProp/update_dueling/eval_net/l1/w1/ApplyRMSPropApplyRMSPropdueling/eval_net/l1/w1,dueling/train/dueling/eval_net/l1/w1/RMSProp.dueling/train/dueling/eval_net/l1/w1/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonRdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
@dueling/train/RMSProp/update_dueling/eval_net/l1/b1/ApplyRMSPropApplyRMSPropdueling/eval_net/l1/b1,dueling/train/dueling/eval_net/l1/b1/RMSProp.dueling/train/dueling/eval_net/l1/b1/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonOdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
Cdueling/train/RMSProp/update_dueling/eval_net/Value/w2/ApplyRMSPropApplyRMSPropdueling/eval_net/Value/w2/dueling/train/dueling/eval_net/Value/w2/RMSProp1dueling/train/dueling/eval_net/Value/w2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonUdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:*
use_locking( 
�
Cdueling/train/RMSProp/update_dueling/eval_net/Value/b2/ApplyRMSPropApplyRMSPropdueling/eval_net/Value/b2/dueling/train/dueling/eval_net/Value/b2/RMSProp1dueling/train/dueling/eval_net/Value/b2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonRdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
Gdueling/train/RMSProp/update_dueling/eval_net/Advantage/w2/ApplyRMSPropApplyRMSPropdueling/eval_net/Advantage/w23dueling/train/dueling/eval_net/Advantage/w2/RMSProp5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonYdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
Gdueling/train/RMSProp/update_dueling/eval_net/Advantage/b2/ApplyRMSPropApplyRMSPropdueling/eval_net/Advantage/b23dueling/train/dueling/eval_net/Advantage/b2/RMSProp5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonVdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency_1*
use_locking( *
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
_output_shapes

:
�
dueling/train/RMSPropNoOpH^dueling/train/RMSProp/update_dueling/eval_net/Advantage/b2/ApplyRMSPropH^dueling/train/RMSProp/update_dueling/eval_net/Advantage/w2/ApplyRMSPropD^dueling/train/RMSProp/update_dueling/eval_net/Value/b2/ApplyRMSPropD^dueling/train/RMSProp/update_dueling/eval_net/Value/w2/ApplyRMSPropA^dueling/train/RMSProp/update_dueling/eval_net/l1/b1/ApplyRMSPropA^dueling/train/RMSProp/update_dueling/eval_net/l1/w1/ApplyRMSProp
m

dueling/s_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
8dueling/target_net/l1/w1/Initializer/random_normal/shapeConst*+
_class!
loc:@dueling/target_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
7dueling/target_net/l1/w1/Initializer/random_normal/meanConst*+
_class!
loc:@dueling/target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
9dueling/target_net/l1/w1/Initializer/random_normal/stddevConst*+
_class!
loc:@dueling/target_net/l1/w1*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Gdueling/target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8dueling/target_net/l1/w1/Initializer/random_normal/shape*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
seed2�*
dtype0*
_output_shapes

:*

seed
�
6dueling/target_net/l1/w1/Initializer/random_normal/mulMulGdueling/target_net/l1/w1/Initializer/random_normal/RandomStandardNormal9dueling/target_net/l1/w1/Initializer/random_normal/stddev*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
_output_shapes

:
�
2dueling/target_net/l1/w1/Initializer/random_normalAdd6dueling/target_net/l1/w1/Initializer/random_normal/mul7dueling/target_net/l1/w1/Initializer/random_normal/mean*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
_output_shapes

:
�
dueling/target_net/l1/w1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dueling/target_net/l1/w1*
	container *
shape
:
�
dueling/target_net/l1/w1/AssignAssigndueling/target_net/l1/w12dueling/target_net/l1/w1/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
validate_shape(
�
dueling/target_net/l1/w1/readIdentitydueling/target_net/l1/w1*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
_output_shapes

:
�
*dueling/target_net/l1/b1/Initializer/ConstConst*+
_class!
loc:@dueling/target_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/target_net/l1/b1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dueling/target_net/l1/b1
�
dueling/target_net/l1/b1/AssignAssigndueling/target_net/l1/b1*dueling/target_net/l1/b1/Initializer/Const*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/b1*
validate_shape(
�
dueling/target_net/l1/b1/readIdentitydueling/target_net/l1/b1*
T0*+
_class!
loc:@dueling/target_net/l1/b1*
_output_shapes

:
�
dueling/target_net/l1/MatMulMatMul
dueling/s_dueling/target_net/l1/w1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dueling/target_net/l1/addAdddueling/target_net/l1/MatMuldueling/target_net/l1/b1/read*'
_output_shapes
:���������*
T0
o
dueling/target_net/l1/ReluReludueling/target_net/l1/add*
T0*'
_output_shapes
:���������
�
;dueling/target_net/Value/w2/Initializer/random_normal/shapeConst*.
_class$
" loc:@dueling/target_net/Value/w2*
valueB"      *
dtype0*
_output_shapes
:
�
:dueling/target_net/Value/w2/Initializer/random_normal/meanConst*.
_class$
" loc:@dueling/target_net/Value/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<dueling/target_net/Value/w2/Initializer/random_normal/stddevConst*.
_class$
" loc:@dueling/target_net/Value/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Jdueling/target_net/Value/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;dueling/target_net/Value/w2/Initializer/random_normal/shape*.
_class$
" loc:@dueling/target_net/Value/w2*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0
�
9dueling/target_net/Value/w2/Initializer/random_normal/mulMulJdueling/target_net/Value/w2/Initializer/random_normal/RandomStandardNormal<dueling/target_net/Value/w2/Initializer/random_normal/stddev*
_output_shapes

:*
T0*.
_class$
" loc:@dueling/target_net/Value/w2
�
5dueling/target_net/Value/w2/Initializer/random_normalAdd9dueling/target_net/Value/w2/Initializer/random_normal/mul:dueling/target_net/Value/w2/Initializer/random_normal/mean*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
_output_shapes

:
�
dueling/target_net/Value/w2
VariableV2*.
_class$
" loc:@dueling/target_net/Value/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"dueling/target_net/Value/w2/AssignAssigndueling/target_net/Value/w25dueling/target_net/Value/w2/Initializer/random_normal*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
validate_shape(*
_output_shapes

:
�
 dueling/target_net/Value/w2/readIdentitydueling/target_net/Value/w2*
_output_shapes

:*
T0*.
_class$
" loc:@dueling/target_net/Value/w2
�
-dueling/target_net/Value/b2/Initializer/ConstConst*.
_class$
" loc:@dueling/target_net/Value/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/target_net/Value/b2
VariableV2*.
_class$
" loc:@dueling/target_net/Value/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"dueling/target_net/Value/b2/AssignAssigndueling/target_net/Value/b2-dueling/target_net/Value/b2/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
validate_shape(*
_output_shapes

:
�
 dueling/target_net/Value/b2/readIdentitydueling/target_net/Value/b2*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
_output_shapes

:
�
dueling/target_net/Value/MatMulMatMuldueling/target_net/l1/Relu dueling/target_net/Value/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/target_net/Value/addAdddueling/target_net/Value/MatMul dueling/target_net/Value/b2/read*'
_output_shapes
:���������*
T0
�
?dueling/target_net/Advantage/w2/Initializer/random_normal/shapeConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB"      *
dtype0*
_output_shapes
:
�
>dueling/target_net/Advantage/w2/Initializer/random_normal/meanConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@dueling/target_net/Advantage/w2/Initializer/random_normal/stddevConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Ndueling/target_net/Advantage/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?dueling/target_net/Advantage/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
seed2�
�
=dueling/target_net/Advantage/w2/Initializer/random_normal/mulMulNdueling/target_net/Advantage/w2/Initializer/random_normal/RandomStandardNormal@dueling/target_net/Advantage/w2/Initializer/random_normal/stddev*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
_output_shapes

:
�
9dueling/target_net/Advantage/w2/Initializer/random_normalAdd=dueling/target_net/Advantage/w2/Initializer/random_normal/mul>dueling/target_net/Advantage/w2/Initializer/random_normal/mean*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
_output_shapes

:
�
dueling/target_net/Advantage/w2
VariableV2*
shared_name *2
_class(
&$loc:@dueling/target_net/Advantage/w2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
&dueling/target_net/Advantage/w2/AssignAssigndueling/target_net/Advantage/w29dueling/target_net/Advantage/w2/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
validate_shape(
�
$dueling/target_net/Advantage/w2/readIdentitydueling/target_net/Advantage/w2*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
_output_shapes

:
�
1dueling/target_net/Advantage/b2/Initializer/ConstConst*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/target_net/Advantage/b2
VariableV2*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
&dueling/target_net/Advantage/b2/AssignAssigndueling/target_net/Advantage/b21dueling/target_net/Advantage/b2/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
$dueling/target_net/Advantage/b2/readIdentitydueling/target_net/Advantage/b2*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
_output_shapes

:*
T0
�
#dueling/target_net/Advantage/MatMulMatMuldueling/target_net/l1/Relu$dueling/target_net/Advantage/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
 dueling/target_net/Advantage/addAdd#dueling/target_net/Advantage/MatMul$dueling/target_net/Advantage/b2/read*
T0*'
_output_shapes
:���������
m
+dueling/target_net/Q/Mean/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dueling/target_net/Q/MeanMean dueling/target_net/Advantage/add+dueling/target_net/Q/Mean/reduction_indices*
T0*'
_output_shapes
:���������*
	keep_dims(*

Tidx0
�
dueling/target_net/Q/subSub dueling/target_net/Advantage/adddueling/target_net/Q/Mean*
T0*'
_output_shapes
:���������
�
dueling/target_net/Q/addAdddueling/target_net/Value/adddueling/target_net/Q/sub*
T0*'
_output_shapes
:���������
�
dueling/AssignAssignnatural/target_net/l1/w1natural/eval_net/l1/w1/read*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:
�
dueling/Assign_1Assignnatural/target_net/l1/b1natural/eval_net/l1/b1/read*+
_class!
loc:@natural/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
dueling/Assign_2Assignnatural/target_net/Q/w2natural/eval_net/Q/w2/read*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/w2*
validate_shape(*
_output_shapes

:
�
dueling/Assign_3Assignnatural/target_net/Q/b2natural/eval_net/Q/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/b2
�
dueling/Assign_4Assigndueling/target_net/l1/w1dueling/eval_net/l1/w1/read*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/Assign_5Assigndueling/target_net/l1/b1dueling/eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/b1
�
dueling/Assign_6Assigndueling/target_net/Value/w2dueling/eval_net/Value/w2/read*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
validate_shape(*
_output_shapes

:
�
dueling/Assign_7Assigndueling/target_net/Value/b2dueling/eval_net/Value/b2/read*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
validate_shape(*
_output_shapes

:
�
dueling/Assign_8Assigndueling/target_net/Advantage/w2"dueling/eval_net/Advantage/w2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2
�
dueling/Assign_9Assigndueling/target_net/Advantage/b2"dueling/eval_net/Advantage/b2/read*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
validate_shape(*
_output_shapes

:*
use_locking("$���=�     �0��	_�Uښ�AJ��
��
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
ApplyRMSProp
var"T�

ms"T�
mom"T�
lr"T
rho"T
momentum"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( 
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �*1.9.02v1.9.0-0-g25c197e023��
l
	natural/sPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
s
natural/Q_targetPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
6natural/eval_net/l1/w1/Initializer/random_normal/shapeConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
5natural/eval_net/l1/w1/Initializer/random_normal/meanConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7natural/eval_net/l1/w1/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *)
_class
loc:@natural/eval_net/l1/w1*
valueB
 *���>
�
Enatural/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6natural/eval_net/l1/w1/Initializer/random_normal/shape*
_output_shapes

:*

seed*
T0*)
_class
loc:@natural/eval_net/l1/w1*
seed2*
dtype0
�
4natural/eval_net/l1/w1/Initializer/random_normal/mulMulEnatural/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal7natural/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
0natural/eval_net/l1/w1/Initializer/random_normalAdd4natural/eval_net/l1/w1/Initializer/random_normal/mul5natural/eval_net/l1/w1/Initializer/random_normal/mean*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
natural/eval_net/l1/w1
VariableV2*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/eval_net/l1/w1/AssignAssignnatural/eval_net/l1/w10natural/eval_net/l1/w1/Initializer/random_normal*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/w1*
validate_shape(*
_output_shapes

:
�
natural/eval_net/l1/w1/readIdentitynatural/eval_net/l1/w1*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
(natural/eval_net/l1/b1/Initializer/ConstConst*
_output_shapes

:*)
_class
loc:@natural/eval_net/l1/b1*
valueB*���=*
dtype0
�
natural/eval_net/l1/b1
VariableV2*)
_class
loc:@natural/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
natural/eval_net/l1/b1/AssignAssignnatural/eval_net/l1/b1(natural/eval_net/l1/b1/Initializer/Const*
_output_shapes

:*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(
�
natural/eval_net/l1/b1/readIdentitynatural/eval_net/l1/b1*
T0*)
_class
loc:@natural/eval_net/l1/b1*
_output_shapes

:
�
natural/eval_net/l1/MatMulMatMul	natural/snatural/eval_net/l1/w1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
natural/eval_net/l1/addAddnatural/eval_net/l1/MatMulnatural/eval_net/l1/b1/read*
T0*'
_output_shapes
:���������
k
natural/eval_net/l1/ReluRelunatural/eval_net/l1/add*
T0*'
_output_shapes
:���������
�
5natural/eval_net/Q/w2/Initializer/random_normal/shapeConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB"      *
dtype0*
_output_shapes
:
�
4natural/eval_net/Q/w2/Initializer/random_normal/meanConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6natural/eval_net/Q/w2/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: *(
_class
loc:@natural/eval_net/Q/w2*
valueB
 *���>
�
Dnatural/eval_net/Q/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal5natural/eval_net/Q/w2/Initializer/random_normal/shape*
seed2*
dtype0*
_output_shapes

:*

seed*
T0*(
_class
loc:@natural/eval_net/Q/w2
�
3natural/eval_net/Q/w2/Initializer/random_normal/mulMulDnatural/eval_net/Q/w2/Initializer/random_normal/RandomStandardNormal6natural/eval_net/Q/w2/Initializer/random_normal/stddev*
_output_shapes

:*
T0*(
_class
loc:@natural/eval_net/Q/w2
�
/natural/eval_net/Q/w2/Initializer/random_normalAdd3natural/eval_net/Q/w2/Initializer/random_normal/mul4natural/eval_net/Q/w2/Initializer/random_normal/mean*
T0*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:
�
natural/eval_net/Q/w2
VariableV2*(
_class
loc:@natural/eval_net/Q/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
natural/eval_net/Q/w2/AssignAssignnatural/eval_net/Q/w2/natural/eval_net/Q/w2/Initializer/random_normal*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(*
_output_shapes

:
�
natural/eval_net/Q/w2/readIdentitynatural/eval_net/Q/w2*
_output_shapes

:*
T0*(
_class
loc:@natural/eval_net/Q/w2
�
'natural/eval_net/Q/b2/Initializer/ConstConst*
dtype0*
_output_shapes

:*(
_class
loc:@natural/eval_net/Q/b2*
valueB*���=
�
natural/eval_net/Q/b2
VariableV2*(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
natural/eval_net/Q/b2/AssignAssignnatural/eval_net/Q/b2'natural/eval_net/Q/b2/Initializer/Const*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:
�
natural/eval_net/Q/b2/readIdentitynatural/eval_net/Q/b2*
_output_shapes

:*
T0*(
_class
loc:@natural/eval_net/Q/b2
�
natural/eval_net/Q/MatMulMatMulnatural/eval_net/l1/Relunatural/eval_net/Q/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
natural/eval_net/Q/addAddnatural/eval_net/Q/MatMulnatural/eval_net/Q/b2/read*'
_output_shapes
:���������*
T0
�
natural/loss/SquaredDifferenceSquaredDifferencenatural/Q_targetnatural/eval_net/Q/add*
T0*'
_output_shapes
:���������
c
natural/loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
natural/loss/MeanMeannatural/loss/SquaredDifferencenatural/loss/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
`
natural/train/gradients/ShapeConst*
_output_shapes
: *
valueB *
dtype0
f
!natural/train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
natural/train/gradients/FillFillnatural/train/gradients/Shape!natural/train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
<natural/train/gradients/natural/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
6natural/train/gradients/natural/loss/Mean_grad/ReshapeReshapenatural/train/gradients/Fill<natural/train/gradients/natural/loss/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
�
4natural/train/gradients/natural/loss/Mean_grad/ShapeShapenatural/loss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
3natural/train/gradients/natural/loss/Mean_grad/TileTile6natural/train/gradients/natural/loss/Mean_grad/Reshape4natural/train/gradients/natural/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
6natural/train/gradients/natural/loss/Mean_grad/Shape_1Shapenatural/loss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
y
6natural/train/gradients/natural/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
~
4natural/train/gradients/natural/loss/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
3natural/train/gradients/natural/loss/Mean_grad/ProdProd6natural/train/gradients/natural/loss/Mean_grad/Shape_14natural/train/gradients/natural/loss/Mean_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
6natural/train/gradients/natural/loss/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
5natural/train/gradients/natural/loss/Mean_grad/Prod_1Prod6natural/train/gradients/natural/loss/Mean_grad/Shape_26natural/train/gradients/natural/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
z
8natural/train/gradients/natural/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6natural/train/gradients/natural/loss/Mean_grad/MaximumMaximum5natural/train/gradients/natural/loss/Mean_grad/Prod_18natural/train/gradients/natural/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
7natural/train/gradients/natural/loss/Mean_grad/floordivFloorDiv3natural/train/gradients/natural/loss/Mean_grad/Prod6natural/train/gradients/natural/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
3natural/train/gradients/natural/loss/Mean_grad/CastCast7natural/train/gradients/natural/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
6natural/train/gradients/natural/loss/Mean_grad/truedivRealDiv3natural/train/gradients/natural/loss/Mean_grad/Tile3natural/train/gradients/natural/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/ShapeShapenatural/Q_target*
T0*
out_type0*
_output_shapes
:
�
Cnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1Shapenatural/eval_net/Q/add*
T0*
out_type0*
_output_shapes
:
�
Qnatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsAnatural/train/gradients/natural/loss/SquaredDifference_grad/ShapeCnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bnatural/train/gradients/natural/loss/SquaredDifference_grad/scalarConst7^natural/train/gradients/natural/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/mulMulBnatural/train/gradients/natural/loss/SquaredDifference_grad/scalar6natural/train/gradients/natural/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/subSubnatural/Q_targetnatural/eval_net/Q/add7^natural/train/gradients/natural/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Mul?natural/train/gradients/natural/loss/SquaredDifference_grad/mul?natural/train/gradients/natural/loss/SquaredDifference_grad/sub*
T0*'
_output_shapes
:���������
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/SumSumAnatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Qnatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cnatural/train/gradients/natural/loss/SquaredDifference_grad/ReshapeReshape?natural/train/gradients/natural/loss/SquaredDifference_grad/SumAnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
Anatural/train/gradients/natural/loss/SquaredDifference_grad/Sum_1SumAnatural/train/gradients/natural/loss/SquaredDifference_grad/mul_1Snatural/train/gradients/natural/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Enatural/train/gradients/natural/loss/SquaredDifference_grad/Reshape_1ReshapeAnatural/train/gradients/natural/loss/SquaredDifference_grad/Sum_1Cnatural/train/gradients/natural/loss/SquaredDifference_grad/Shape_1*
Tshape0*'
_output_shapes
:���������*
T0
�
?natural/train/gradients/natural/loss/SquaredDifference_grad/NegNegEnatural/train/gradients/natural/loss/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:���������
�
Lnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_depsNoOp@^natural/train/gradients/natural/loss/SquaredDifference_grad/NegD^natural/train/gradients/natural/loss/SquaredDifference_grad/Reshape
�
Tnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependencyIdentityCnatural/train/gradients/natural/loss/SquaredDifference_grad/ReshapeM^natural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_deps*
T0*V
_classL
JHloc:@natural/train/gradients/natural/loss/SquaredDifference_grad/Reshape*'
_output_shapes
:���������
�
Vnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Identity?natural/train/gradients/natural/loss/SquaredDifference_grad/NegM^natural/train/gradients/natural/loss/SquaredDifference_grad/tuple/group_deps*R
_classH
FDloc:@natural/train/gradients/natural/loss/SquaredDifference_grad/Neg*'
_output_shapes
:���������*
T0
�
9natural/train/gradients/natural/eval_net/Q/add_grad/ShapeShapenatural/eval_net/Q/MatMul*
T0*
out_type0*
_output_shapes
:
�
;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1Const*
_output_shapes
:*
valueB"      *
dtype0
�
Inatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgsBroadcastGradientArgs9natural/train/gradients/natural/eval_net/Q/add_grad/Shape;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7natural/train/gradients/natural/eval_net/Q/add_grad/SumSumVnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Inatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
;natural/train/gradients/natural/eval_net/Q/add_grad/ReshapeReshape7natural/train/gradients/natural/eval_net/Q/add_grad/Sum9natural/train/gradients/natural/eval_net/Q/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9natural/train/gradients/natural/eval_net/Q/add_grad/Sum_1SumVnatural/train/gradients/natural/loss/SquaredDifference_grad/tuple/control_dependency_1Knatural/train/gradients/natural/eval_net/Q/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
=natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1Reshape9natural/train/gradients/natural/eval_net/Q/add_grad/Sum_1;natural/train/gradients/natural/eval_net/Q/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Dnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_depsNoOp<^natural/train/gradients/natural/eval_net/Q/add_grad/Reshape>^natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1
�
Lnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependencyIdentity;natural/train/gradients/natural/eval_net/Q/add_grad/ReshapeE^natural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@natural/train/gradients/natural/eval_net/Q/add_grad/Reshape
�
Nnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency_1Identity=natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1E^natural/train/gradients/natural/eval_net/Q/add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@natural/train/gradients/natural/eval_net/Q/add_grad/Reshape_1*
_output_shapes

:
�
=natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMulMatMulLnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependencynatural/eval_net/Q/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
?natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1MatMulnatural/eval_net/l1/ReluLnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
Gnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_depsNoOp>^natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul@^natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1
�
Onatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependencyIdentity=natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMulH^natural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Qnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependency_1Identity?natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1H^natural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@natural/train/gradients/natural/eval_net/Q/MatMul_grad/MatMul_1*
_output_shapes

:
�
>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradReluGradOnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependencynatural/eval_net/l1/Relu*'
_output_shapes
:���������*
T0
�
:natural/train/gradients/natural/eval_net/l1/add_grad/ShapeShapenatural/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
�
<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Jnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:natural/train/gradients/natural/eval_net/l1/add_grad/Shape<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
8natural/train/gradients/natural/eval_net/l1/add_grad/SumSum>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradJnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<natural/train/gradients/natural/eval_net/l1/add_grad/ReshapeReshape8natural/train/gradients/natural/eval_net/l1/add_grad/Sum:natural/train/gradients/natural/eval_net/l1/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
:natural/train/gradients/natural/eval_net/l1/add_grad/Sum_1Sum>natural/train/gradients/natural/eval_net/l1/Relu_grad/ReluGradLnatural/train/gradients/natural/eval_net/l1/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
>natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1Reshape:natural/train/gradients/natural/eval_net/l1/add_grad/Sum_1<natural/train/gradients/natural/eval_net/l1/add_grad/Shape_1*
_output_shapes

:*
T0*
Tshape0
�
Enatural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_depsNoOp=^natural/train/gradients/natural/eval_net/l1/add_grad/Reshape?^natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1
�
Mnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependencyIdentity<natural/train/gradients/natural/eval_net/l1/add_grad/ReshapeF^natural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_deps*
T0*O
_classE
CAloc:@natural/train/gradients/natural/eval_net/l1/add_grad/Reshape*'
_output_shapes
:���������
�
Onatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency_1Identity>natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1F^natural/train/gradients/natural/eval_net/l1/add_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@natural/train/gradients/natural/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:
�
>natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulMatMulMnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependencynatural/eval_net/l1/w1/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1MatMul	natural/sMnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
Hnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_depsNoOp?^natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulA^natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1
�
Pnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity>natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMulI^natural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Rnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1I^natural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*S
_classI
GEloc:@natural/train/gradients/natural/eval_net/l1/MatMul_grad/MatMul_1
�
=natural/train/natural/eval_net/l1/w1/RMSProp/Initializer/onesConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB*  �?*
dtype0*
_output_shapes

:
�
,natural/train/natural/eval_net/l1/w1/RMSProp
VariableV2*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
3natural/train/natural/eval_net/l1/w1/RMSProp/AssignAssign,natural/train/natural/eval_net/l1/w1/RMSProp=natural/train/natural/eval_net/l1/w1/RMSProp/Initializer/ones*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/w1
�
1natural/train/natural/eval_net/l1/w1/RMSProp/readIdentity,natural/train/natural/eval_net/l1/w1/RMSProp*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/w1
�
@natural/train/natural/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@natural/eval_net/l1/w1*
valueB*    *
dtype0*
_output_shapes

:
�
.natural/train/natural/eval_net/l1/w1/RMSProp_1
VariableV2*
shared_name *)
_class
loc:@natural/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
5natural/train/natural/eval_net/l1/w1/RMSProp_1/AssignAssign.natural/train/natural/eval_net/l1/w1/RMSProp_1@natural/train/natural/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/w1*
validate_shape(
�
3natural/train/natural/eval_net/l1/w1/RMSProp_1/readIdentity.natural/train/natural/eval_net/l1/w1/RMSProp_1*
T0*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:
�
=natural/train/natural/eval_net/l1/b1/RMSProp/Initializer/onesConst*
_output_shapes

:*)
_class
loc:@natural/eval_net/l1/b1*
valueB*  �?*
dtype0
�
,natural/train/natural/eval_net/l1/b1/RMSProp
VariableV2*)
_class
loc:@natural/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
3natural/train/natural/eval_net/l1/b1/RMSProp/AssignAssign,natural/train/natural/eval_net/l1/b1/RMSProp=natural/train/natural/eval_net/l1/b1/RMSProp/Initializer/ones*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
1natural/train/natural/eval_net/l1/b1/RMSProp/readIdentity,natural/train/natural/eval_net/l1/b1/RMSProp*
T0*)
_class
loc:@natural/eval_net/l1/b1*
_output_shapes

:
�
@natural/train/natural/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@natural/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
.natural/train/natural/eval_net/l1/b1/RMSProp_1
VariableV2*)
_class
loc:@natural/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
5natural/train/natural/eval_net/l1/b1/RMSProp_1/AssignAssign.natural/train/natural/eval_net/l1/b1/RMSProp_1@natural/train/natural/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@natural/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
3natural/train/natural/eval_net/l1/b1/RMSProp_1/readIdentity.natural/train/natural/eval_net/l1/b1/RMSProp_1*
_output_shapes

:*
T0*)
_class
loc:@natural/eval_net/l1/b1
�
<natural/train/natural/eval_net/Q/w2/RMSProp/Initializer/onesConst*(
_class
loc:@natural/eval_net/Q/w2*
valueB*  �?*
dtype0*
_output_shapes

:
�
+natural/train/natural/eval_net/Q/w2/RMSProp
VariableV2*
shared_name *(
_class
loc:@natural/eval_net/Q/w2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
2natural/train/natural/eval_net/Q/w2/RMSProp/AssignAssign+natural/train/natural/eval_net/Q/w2/RMSProp<natural/train/natural/eval_net/Q/w2/RMSProp/Initializer/ones*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(*
_output_shapes

:
�
0natural/train/natural/eval_net/Q/w2/RMSProp/readIdentity+natural/train/natural/eval_net/Q/w2/RMSProp*
T0*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:
�
?natural/train/natural/eval_net/Q/w2/RMSProp_1/Initializer/zerosConst*
_output_shapes

:*(
_class
loc:@natural/eval_net/Q/w2*
valueB*    *
dtype0
�
-natural/train/natural/eval_net/Q/w2/RMSProp_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/w2*
	container *
shape
:
�
4natural/train/natural/eval_net/Q/w2/RMSProp_1/AssignAssign-natural/train/natural/eval_net/Q/w2/RMSProp_1?natural/train/natural/eval_net/Q/w2/RMSProp_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/w2*
validate_shape(
�
2natural/train/natural/eval_net/Q/w2/RMSProp_1/readIdentity-natural/train/natural/eval_net/Q/w2/RMSProp_1*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:*
T0
�
<natural/train/natural/eval_net/Q/b2/RMSProp/Initializer/onesConst*
_output_shapes

:*(
_class
loc:@natural/eval_net/Q/b2*
valueB*  �?*
dtype0
�
+natural/train/natural/eval_net/Q/b2/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:
�
2natural/train/natural/eval_net/Q/b2/RMSProp/AssignAssign+natural/train/natural/eval_net/Q/b2/RMSProp<natural/train/natural/eval_net/Q/b2/RMSProp/Initializer/ones*
use_locking(*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:
�
0natural/train/natural/eval_net/Q/b2/RMSProp/readIdentity+natural/train/natural/eval_net/Q/b2/RMSProp*
T0*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:
�
?natural/train/natural/eval_net/Q/b2/RMSProp_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*(
_class
loc:@natural/eval_net/Q/b2*
valueB*    
�
-natural/train/natural/eval_net/Q/b2/RMSProp_1
VariableV2*
shared_name *(
_class
loc:@natural/eval_net/Q/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
4natural/train/natural/eval_net/Q/b2/RMSProp_1/AssignAssign-natural/train/natural/eval_net/Q/b2/RMSProp_1?natural/train/natural/eval_net/Q/b2/RMSProp_1/Initializer/zeros*
T0*(
_class
loc:@natural/eval_net/Q/b2*
validate_shape(*
_output_shapes

:*
use_locking(
�
2natural/train/natural/eval_net/Q/b2/RMSProp_1/readIdentity-natural/train/natural/eval_net/Q/b2/RMSProp_1*
T0*(
_class
loc:@natural/eval_net/Q/b2*
_output_shapes

:
h
#natural/train/RMSProp/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
`
natural/train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
c
natural/train/RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
natural/train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
@natural/train/RMSProp/update_natural/eval_net/l1/w1/ApplyRMSPropApplyRMSPropnatural/eval_net/l1/w1,natural/train/natural/eval_net/l1/w1/RMSProp.natural/train/natural/eval_net/l1/w1/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonRnatural/train/gradients/natural/eval_net/l1/MatMul_grad/tuple/control_dependency_1*)
_class
loc:@natural/eval_net/l1/w1*
_output_shapes

:*
use_locking( *
T0
�
@natural/train/RMSProp/update_natural/eval_net/l1/b1/ApplyRMSPropApplyRMSPropnatural/eval_net/l1/b1,natural/train/natural/eval_net/l1/b1/RMSProp.natural/train/natural/eval_net/l1/b1/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonOnatural/train/gradients/natural/eval_net/l1/add_grad/tuple/control_dependency_1*
T0*)
_class
loc:@natural/eval_net/l1/b1*
_output_shapes

:*
use_locking( 
�
?natural/train/RMSProp/update_natural/eval_net/Q/w2/ApplyRMSPropApplyRMSPropnatural/eval_net/Q/w2+natural/train/natural/eval_net/Q/w2/RMSProp-natural/train/natural/eval_net/Q/w2/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonQnatural/train/gradients/natural/eval_net/Q/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*(
_class
loc:@natural/eval_net/Q/w2*
_output_shapes

:
�
?natural/train/RMSProp/update_natural/eval_net/Q/b2/ApplyRMSPropApplyRMSPropnatural/eval_net/Q/b2+natural/train/natural/eval_net/Q/b2/RMSProp-natural/train/natural/eval_net/Q/b2/RMSProp_1#natural/train/RMSProp/learning_ratenatural/train/RMSProp/decaynatural/train/RMSProp/momentumnatural/train/RMSProp/epsilonNnatural/train/gradients/natural/eval_net/Q/add_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*(
_class
loc:@natural/eval_net/Q/b2
�
natural/train/RMSPropNoOp@^natural/train/RMSProp/update_natural/eval_net/Q/b2/ApplyRMSProp@^natural/train/RMSProp/update_natural/eval_net/Q/w2/ApplyRMSPropA^natural/train/RMSProp/update_natural/eval_net/l1/b1/ApplyRMSPropA^natural/train/RMSProp/update_natural/eval_net/l1/w1/ApplyRMSProp
m

natural/s_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
8natural/target_net/l1/w1/Initializer/random_normal/shapeConst*
_output_shapes
:*+
_class!
loc:@natural/target_net/l1/w1*
valueB"      *
dtype0
�
7natural/target_net/l1/w1/Initializer/random_normal/meanConst*+
_class!
loc:@natural/target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
9natural/target_net/l1/w1/Initializer/random_normal/stddevConst*+
_class!
loc:@natural/target_net/l1/w1*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Gnatural/target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8natural/target_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*+
_class!
loc:@natural/target_net/l1/w1*
seed2�
�
6natural/target_net/l1/w1/Initializer/random_normal/mulMulGnatural/target_net/l1/w1/Initializer/random_normal/RandomStandardNormal9natural/target_net/l1/w1/Initializer/random_normal/stddev*+
_class!
loc:@natural/target_net/l1/w1*
_output_shapes

:*
T0
�
2natural/target_net/l1/w1/Initializer/random_normalAdd6natural/target_net/l1/w1/Initializer/random_normal/mul7natural/target_net/l1/w1/Initializer/random_normal/mean*
_output_shapes

:*
T0*+
_class!
loc:@natural/target_net/l1/w1
�
natural/target_net/l1/w1
VariableV2*
_output_shapes

:*
shared_name *+
_class!
loc:@natural/target_net/l1/w1*
	container *
shape
:*
dtype0
�
natural/target_net/l1/w1/AssignAssignnatural/target_net/l1/w12natural/target_net/l1/w1/Initializer/random_normal*
T0*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
natural/target_net/l1/w1/readIdentitynatural/target_net/l1/w1*+
_class!
loc:@natural/target_net/l1/w1*
_output_shapes

:*
T0
�
*natural/target_net/l1/b1/Initializer/ConstConst*
dtype0*
_output_shapes

:*+
_class!
loc:@natural/target_net/l1/b1*
valueB*���=
�
natural/target_net/l1/b1
VariableV2*
shared_name *+
_class!
loc:@natural/target_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
natural/target_net/l1/b1/AssignAssignnatural/target_net/l1/b1*natural/target_net/l1/b1/Initializer/Const*+
_class!
loc:@natural/target_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
natural/target_net/l1/b1/readIdentitynatural/target_net/l1/b1*+
_class!
loc:@natural/target_net/l1/b1*
_output_shapes

:*
T0
�
natural/target_net/l1/MatMulMatMul
natural/s_natural/target_net/l1/w1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
natural/target_net/l1/addAddnatural/target_net/l1/MatMulnatural/target_net/l1/b1/read*
T0*'
_output_shapes
:���������
o
natural/target_net/l1/ReluRelunatural/target_net/l1/add*
T0*'
_output_shapes
:���������
�
7natural/target_net/Q/w2/Initializer/random_normal/shapeConst**
_class 
loc:@natural/target_net/Q/w2*
valueB"      *
dtype0*
_output_shapes
:
�
6natural/target_net/Q/w2/Initializer/random_normal/meanConst**
_class 
loc:@natural/target_net/Q/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8natural/target_net/Q/w2/Initializer/random_normal/stddevConst*
dtype0*
_output_shapes
: **
_class 
loc:@natural/target_net/Q/w2*
valueB
 *���>
�
Fnatural/target_net/Q/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal7natural/target_net/Q/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0**
_class 
loc:@natural/target_net/Q/w2*
seed2�
�
5natural/target_net/Q/w2/Initializer/random_normal/mulMulFnatural/target_net/Q/w2/Initializer/random_normal/RandomStandardNormal8natural/target_net/Q/w2/Initializer/random_normal/stddev*
T0**
_class 
loc:@natural/target_net/Q/w2*
_output_shapes

:
�
1natural/target_net/Q/w2/Initializer/random_normalAdd5natural/target_net/Q/w2/Initializer/random_normal/mul6natural/target_net/Q/w2/Initializer/random_normal/mean*
_output_shapes

:*
T0**
_class 
loc:@natural/target_net/Q/w2
�
natural/target_net/Q/w2
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@natural/target_net/Q/w2*
	container *
shape
:
�
natural/target_net/Q/w2/AssignAssignnatural/target_net/Q/w21natural/target_net/Q/w2/Initializer/random_normal*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/w2*
validate_shape(
�
natural/target_net/Q/w2/readIdentitynatural/target_net/Q/w2*
_output_shapes

:*
T0**
_class 
loc:@natural/target_net/Q/w2
�
)natural/target_net/Q/b2/Initializer/ConstConst**
_class 
loc:@natural/target_net/Q/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
natural/target_net/Q/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name **
_class 
loc:@natural/target_net/Q/b2*
	container *
shape
:
�
natural/target_net/Q/b2/AssignAssignnatural/target_net/Q/b2)natural/target_net/Q/b2/Initializer/Const**
_class 
loc:@natural/target_net/Q/b2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
natural/target_net/Q/b2/readIdentitynatural/target_net/Q/b2*
T0**
_class 
loc:@natural/target_net/Q/b2*
_output_shapes

:
�
natural/target_net/Q/MatMulMatMulnatural/target_net/l1/Relunatural/target_net/Q/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
natural/target_net/Q/addAddnatural/target_net/Q/MatMulnatural/target_net/Q/b2/read*
T0*'
_output_shapes
:���������
�
natural/AssignAssignnatural/target_net/l1/w1natural/eval_net/l1/w1/read*
T0*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
natural/Assign_1Assignnatural/target_net/l1/b1natural/eval_net/l1/b1/read*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
natural/Assign_2Assignnatural/target_net/Q/w2natural/eval_net/Q/w2/read*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/w2*
validate_shape(*
_output_shapes

:
�
natural/Assign_3Assignnatural/target_net/Q/b2natural/eval_net/Q/b2/read**
_class 
loc:@natural/target_net/Q/b2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
l
	dueling/sPlaceholder*
shape:���������*
dtype0*'
_output_shapes
:���������
s
dueling/Q_targetPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
6dueling/eval_net/l1/w1/Initializer/random_normal/shapeConst*
dtype0*
_output_shapes
:*)
_class
loc:@dueling/eval_net/l1/w1*
valueB"      
�
5dueling/eval_net/l1/w1/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *)
_class
loc:@dueling/eval_net/l1/w1*
valueB
 *    
�
7dueling/eval_net/l1/w1/Initializer/random_normal/stddevConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Edueling/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal6dueling/eval_net/l1/w1/Initializer/random_normal/shape*

seed*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
seed2�*
dtype0*
_output_shapes

:
�
4dueling/eval_net/l1/w1/Initializer/random_normal/mulMulEdueling/eval_net/l1/w1/Initializer/random_normal/RandomStandardNormal7dueling/eval_net/l1/w1/Initializer/random_normal/stddev*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
0dueling/eval_net/l1/w1/Initializer/random_normalAdd4dueling/eval_net/l1/w1/Initializer/random_normal/mul5dueling/eval_net/l1/w1/Initializer/random_normal/mean*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
dueling/eval_net/l1/w1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/w1*
	container *
shape
:
�
dueling/eval_net/l1/w1/AssignAssigndueling/eval_net/l1/w10dueling/eval_net/l1/w1/Initializer/random_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/w1
�
dueling/eval_net/l1/w1/readIdentitydueling/eval_net/l1/w1*
_output_shapes

:*
T0*)
_class
loc:@dueling/eval_net/l1/w1
�
(dueling/eval_net/l1/b1/Initializer/ConstConst*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/eval_net/l1/b1
VariableV2*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
dueling/eval_net/l1/b1/AssignAssigndueling/eval_net/l1/b1(dueling/eval_net/l1/b1/Initializer/Const*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
dueling/eval_net/l1/b1/readIdentitydueling/eval_net/l1/b1*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:*
T0
�
dueling/eval_net/l1/MatMulMatMul	dueling/sdueling/eval_net/l1/w1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/eval_net/l1/addAdddueling/eval_net/l1/MatMuldueling/eval_net/l1/b1/read*
T0*'
_output_shapes
:���������
k
dueling/eval_net/l1/ReluReludueling/eval_net/l1/add*
T0*'
_output_shapes
:���������
�
9dueling/eval_net/Value/w2/Initializer/random_normal/shapeConst*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB"      *
dtype0*
_output_shapes
:
�
8dueling/eval_net/Value/w2/Initializer/random_normal/meanConst*
_output_shapes
: *,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB
 *    *
dtype0
�
:dueling/eval_net/Value/w2/Initializer/random_normal/stddevConst*
_output_shapes
: *,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB
 *���>*
dtype0
�
Hdueling/eval_net/Value/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal9dueling/eval_net/Value/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
seed2�
�
7dueling/eval_net/Value/w2/Initializer/random_normal/mulMulHdueling/eval_net/Value/w2/Initializer/random_normal/RandomStandardNormal:dueling/eval_net/Value/w2/Initializer/random_normal/stddev*
_output_shapes

:*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2
�
3dueling/eval_net/Value/w2/Initializer/random_normalAdd7dueling/eval_net/Value/w2/Initializer/random_normal/mul8dueling/eval_net/Value/w2/Initializer/random_normal/mean*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
dueling/eval_net/Value/w2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/w2*
	container 
�
 dueling/eval_net/Value/w2/AssignAssigndueling/eval_net/Value/w23dueling/eval_net/Value/w2/Initializer/random_normal*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/eval_net/Value/w2/readIdentitydueling/eval_net/Value/w2*
_output_shapes

:*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2
�
+dueling/eval_net/Value/b2/Initializer/ConstConst*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/eval_net/Value/b2
VariableV2*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
 dueling/eval_net/Value/b2/AssignAssigndueling/eval_net/Value/b2+dueling/eval_net/Value/b2/Initializer/Const*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/eval_net/Value/b2/readIdentitydueling/eval_net/Value/b2*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
dueling/eval_net/Value/MatMulMatMuldueling/eval_net/l1/Reludueling/eval_net/Value/w2/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dueling/eval_net/Value/addAdddueling/eval_net/Value/MatMuldueling/eval_net/Value/b2/read*'
_output_shapes
:���������*
T0
�
=dueling/eval_net/Advantage/w2/Initializer/random_normal/shapeConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB"      *
dtype0*
_output_shapes
:
�
<dueling/eval_net/Advantage/w2/Initializer/random_normal/meanConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB
 *    
�
>dueling/eval_net/Advantage/w2/Initializer/random_normal/stddevConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Ldueling/eval_net/Advantage/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal=dueling/eval_net/Advantage/w2/Initializer/random_normal/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2
�
;dueling/eval_net/Advantage/w2/Initializer/random_normal/mulMulLdueling/eval_net/Advantage/w2/Initializer/random_normal/RandomStandardNormal>dueling/eval_net/Advantage/w2/Initializer/random_normal/stddev*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
7dueling/eval_net/Advantage/w2/Initializer/random_normalAdd;dueling/eval_net/Advantage/w2/Initializer/random_normal/mul<dueling/eval_net/Advantage/w2/Initializer/random_normal/mean*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
dueling/eval_net/Advantage/w2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
	container *
shape
:
�
$dueling/eval_net/Advantage/w2/AssignAssigndueling/eval_net/Advantage/w27dueling/eval_net/Advantage/w2/Initializer/random_normal*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
"dueling/eval_net/Advantage/w2/readIdentitydueling/eval_net/Advantage/w2*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
/dueling/eval_net/Advantage/b2/Initializer/ConstConst*
_output_shapes

:*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*���=*
dtype0
�
dueling/eval_net/Advantage/b2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
	container *
shape
:
�
$dueling/eval_net/Advantage/b2/AssignAssigndueling/eval_net/Advantage/b2/dueling/eval_net/Advantage/b2/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
"dueling/eval_net/Advantage/b2/readIdentitydueling/eval_net/Advantage/b2*
_output_shapes

:*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2
�
!dueling/eval_net/Advantage/MatMulMatMuldueling/eval_net/l1/Relu"dueling/eval_net/Advantage/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dueling/eval_net/Advantage/addAdd!dueling/eval_net/Advantage/MatMul"dueling/eval_net/Advantage/b2/read*
T0*'
_output_shapes
:���������
k
)dueling/eval_net/Q/Mean/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
�
dueling/eval_net/Q/MeanMeandueling/eval_net/Advantage/add)dueling/eval_net/Q/Mean/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
dueling/eval_net/Q/subSubdueling/eval_net/Advantage/adddueling/eval_net/Q/Mean*
T0*'
_output_shapes
:���������
�
dueling/eval_net/Q/addAdddueling/eval_net/Value/adddueling/eval_net/Q/sub*
T0*'
_output_shapes
:���������
�
dueling/loss/SquaredDifferenceSquaredDifferencedueling/Q_targetdueling/eval_net/Q/add*'
_output_shapes
:���������*
T0
c
dueling/loss/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
dueling/loss/MeanMeandueling/loss/SquaredDifferencedueling/loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
`
dueling/train/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
f
!dueling/train/gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
dueling/train/gradients/FillFilldueling/train/gradients/Shape!dueling/train/gradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
�
<dueling/train/gradients/dueling/loss/Mean_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
�
6dueling/train/gradients/dueling/loss/Mean_grad/ReshapeReshapedueling/train/gradients/Fill<dueling/train/gradients/dueling/loss/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
�
4dueling/train/gradients/dueling/loss/Mean_grad/ShapeShapedueling/loss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
�
3dueling/train/gradients/dueling/loss/Mean_grad/TileTile6dueling/train/gradients/dueling/loss/Mean_grad/Reshape4dueling/train/gradients/dueling/loss/Mean_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
�
6dueling/train/gradients/dueling/loss/Mean_grad/Shape_1Shapedueling/loss/SquaredDifference*
T0*
out_type0*
_output_shapes
:
y
6dueling/train/gradients/dueling/loss/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
~
4dueling/train/gradients/dueling/loss/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
3dueling/train/gradients/dueling/loss/Mean_grad/ProdProd6dueling/train/gradients/dueling/loss/Mean_grad/Shape_14dueling/train/gradients/dueling/loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
6dueling/train/gradients/dueling/loss/Mean_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
5dueling/train/gradients/dueling/loss/Mean_grad/Prod_1Prod6dueling/train/gradients/dueling/loss/Mean_grad/Shape_26dueling/train/gradients/dueling/loss/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
z
8dueling/train/gradients/dueling/loss/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
6dueling/train/gradients/dueling/loss/Mean_grad/MaximumMaximum5dueling/train/gradients/dueling/loss/Mean_grad/Prod_18dueling/train/gradients/dueling/loss/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
�
7dueling/train/gradients/dueling/loss/Mean_grad/floordivFloorDiv3dueling/train/gradients/dueling/loss/Mean_grad/Prod6dueling/train/gradients/dueling/loss/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
3dueling/train/gradients/dueling/loss/Mean_grad/CastCast7dueling/train/gradients/dueling/loss/Mean_grad/floordiv*

SrcT0*
_output_shapes
: *

DstT0
�
6dueling/train/gradients/dueling/loss/Mean_grad/truedivRealDiv3dueling/train/gradients/dueling/loss/Mean_grad/Tile3dueling/train/gradients/dueling/loss/Mean_grad/Cast*
T0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/ShapeShapedueling/Q_target*
_output_shapes
:*
T0*
out_type0
�
Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1Shapedueling/eval_net/Q/add*
T0*
out_type0*
_output_shapes
:
�
Qdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgsAdueling/train/gradients/dueling/loss/SquaredDifference_grad/ShapeCdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
Bdueling/train/gradients/dueling/loss/SquaredDifference_grad/scalarConst7^dueling/train/gradients/dueling/loss/Mean_grad/truediv*
valueB
 *   @*
dtype0*
_output_shapes
: 
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/mulMulBdueling/train/gradients/dueling/loss/SquaredDifference_grad/scalar6dueling/train/gradients/dueling/loss/Mean_grad/truediv*'
_output_shapes
:���������*
T0
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/subSubdueling/Q_targetdueling/eval_net/Q/add7^dueling/train/gradients/dueling/loss/Mean_grad/truediv*
T0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Mul?dueling/train/gradients/dueling/loss/SquaredDifference_grad/mul?dueling/train/gradients/dueling/loss/SquaredDifference_grad/sub*'
_output_shapes
:���������*
T0
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/SumSumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Qdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/ReshapeReshape?dueling/train/gradients/dueling/loss/SquaredDifference_grad/SumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/loss/SquaredDifference_grad/Sum_1SumAdueling/train/gradients/dueling/loss/SquaredDifference_grad/mul_1Sdueling/train/gradients/dueling/loss/SquaredDifference_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Edueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape_1ReshapeAdueling/train/gradients/dueling/loss/SquaredDifference_grad/Sum_1Cdueling/train/gradients/dueling/loss/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
?dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegNegEdueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape_1*'
_output_shapes
:���������*
T0
�
Ldueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_depsNoOp@^dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegD^dueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape
�
Tdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependencyIdentityCdueling/train/gradients/dueling/loss/SquaredDifference_grad/ReshapeM^dueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*V
_classL
JHloc:@dueling/train/gradients/dueling/loss/SquaredDifference_grad/Reshape
�
Vdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Identity?dueling/train/gradients/dueling/loss/SquaredDifference_grad/NegM^dueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/group_deps*
T0*R
_classH
FDloc:@dueling/train/gradients/dueling/loss/SquaredDifference_grad/Neg*'
_output_shapes
:���������
�
9dueling/train/gradients/dueling/eval_net/Q/add_grad/ShapeShapedueling/eval_net/Value/add*
T0*
out_type0*
_output_shapes
:
�
;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1Shapedueling/eval_net/Q/sub*
_output_shapes
:*
T0*
out_type0
�
Idueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgsBroadcastGradientArgs9dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7dueling/train/gradients/dueling/eval_net/Q/add_grad/SumSumVdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Idueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
;dueling/train/gradients/dueling/eval_net/Q/add_grad/ReshapeReshape7dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum9dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
9dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum_1SumVdueling/train/gradients/dueling/loss/SquaredDifference_grad/tuple/control_dependency_1Kdueling/train/gradients/dueling/eval_net/Q/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
=dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1Reshape9dueling/train/gradients/dueling/eval_net/Q/add_grad/Sum_1;dueling/train/gradients/dueling/eval_net/Q/add_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_depsNoOp<^dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape>^dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1
�
Ldueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyIdentity;dueling/train/gradients/dueling/eval_net/Q/add_grad/ReshapeE^dueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape*'
_output_shapes
:���������
�
Ndueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Identity=dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1E^dueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@dueling/train/gradients/dueling/eval_net/Q/add_grad/Reshape_1*'
_output_shapes
:���������
�
=dueling/train/gradients/dueling/eval_net/Value/add_grad/ShapeShapedueling/eval_net/Value/MatMul*
T0*
out_type0*
_output_shapes
:
�
?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Mdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgsBroadcastGradientArgs=dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
;dueling/train/gradients/dueling/eval_net/Value/add_grad/SumSumLdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyMdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
?dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeReshape;dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum=dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
=dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum_1SumLdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependencyOdueling/train/gradients/dueling/eval_net/Value/add_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
Adueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1Reshape=dueling/train/gradients/dueling/eval_net/Value/add_grad/Sum_1?dueling/train/gradients/dueling/eval_net/Value/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Hdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_depsNoOp@^dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeB^dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1
�
Pdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependencyIdentity?dueling/train/gradients/dueling/eval_net/Value/add_grad/ReshapeI^dueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_deps*
T0*R
_classH
FDloc:@dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape*'
_output_shapes
:���������
�
Rdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency_1IdentityAdueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1I^dueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/group_deps*
_output_shapes

:*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/add_grad/Reshape_1
�
9dueling/train/gradients/dueling/eval_net/Q/sub_grad/ShapeShapedueling/eval_net/Advantage/add*
T0*
out_type0*
_output_shapes
:
�
;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1Shapedueling/eval_net/Q/Mean*
out_type0*
_output_shapes
:*
T0
�
Idueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgsBroadcastGradientArgs9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
7dueling/train/gradients/dueling/eval_net/Q/sub_grad/SumSumNdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Idueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
;dueling/train/gradients/dueling/eval_net/Q/sub_grad/ReshapeReshape7dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum_1SumNdueling/train/gradients/dueling/eval_net/Q/add_grad/tuple/control_dependency_1Kdueling/train/gradients/dueling/eval_net/Q/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
7dueling/train/gradients/dueling/eval_net/Q/sub_grad/NegNeg9dueling/train/gradients/dueling/eval_net/Q/sub_grad/Sum_1*
T0*
_output_shapes
:
�
=dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1Reshape7dueling/train/gradients/dueling/eval_net/Q/sub_grad/Neg;dueling/train/gradients/dueling/eval_net/Q/sub_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
�
Ddueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_depsNoOp<^dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape>^dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1
�
Ldueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependencyIdentity;dueling/train/gradients/dueling/eval_net/Q/sub_grad/ReshapeE^dueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape
�
Ndueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency_1Identity=dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1E^dueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/group_deps*
T0*P
_classF
DBloc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape_1*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulMatMulPdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependencydueling/eval_net/Value/w2/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
Cdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1MatMuldueling/eval_net/l1/ReluPdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency*
T0*
_output_shapes

:*
transpose_a(*
transpose_b( 
�
Kdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_depsNoOpB^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulD^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1
�
Sdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependencyIdentityAdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMulL^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Udueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependency_1IdentityCdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1L^dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/group_deps*V
_classL
JHloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ShapeShapedueling/eval_net/Advantage/add*
T0*
out_type0*
_output_shapes
:
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/SizeConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/addAdd)dueling/eval_net/Q/Mean/reduction_indices9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size*
_output_shapes
: *
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape
�
8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/modFloorMod8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/add9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
: 
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_1Const*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
valueB *
dtype0*
_output_shapes
: 
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/startConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/deltaConst*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/rangeRange@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/start9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Size@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range/delta*

Tidx0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
:
�
?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill/valueConst*
_output_shapes
: *M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/FillFill<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_1?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill/value*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*

index_type0*
_output_shapes
: 
�
Bdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitchDynamicStitch:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/range8dueling/train/gradients/dueling/eval_net/Q/Mean_grad/mod:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Fill*#
_output_shapes
:���������*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
N
�
>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum/yConst*
_output_shapes
: *M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
value	B :*
dtype0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/MaximumMaximumBdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitch>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum/y*#
_output_shapes
:���������*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape
�
=dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordivFloorDiv:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum*
T0*M
_classC
A?loc:@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape*
_output_shapes
:
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ReshapeReshapeNdueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency_1Bdueling/train/gradients/dueling/eval_net/Q/Mean_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/TileTile<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Reshape=dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_2Shapedueling/eval_net/Advantage/add*
_output_shapes
:*
T0*
out_type0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_3Shapedueling/eval_net/Q/Mean*
_output_shapes
:*
T0*
out_type0
�
:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/ProdProd<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_2:dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
;dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod_1Prod<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Shape_3<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1Maximum;dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod_1@dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1/y*
_output_shapes
: *
T0
�
?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv_1FloorDiv9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Prod>dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Maximum_1*
_output_shapes
: *
T0
�
9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/CastCast?dueling/train/gradients/dueling/eval_net/Q/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0
�
<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/truedivRealDiv9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Tile9dueling/train/gradients/dueling/eval_net/Q/Mean_grad/Cast*'
_output_shapes
:���������*
T0
�
dueling/train/gradients/AddNAddNLdueling/train/gradients/dueling/eval_net/Q/sub_grad/tuple/control_dependency<dueling/train/gradients/dueling/eval_net/Q/Mean_grad/truediv*N
_classD
B@loc:@dueling/train/gradients/dueling/eval_net/Q/sub_grad/Reshape*
N*'
_output_shapes
:���������*
T0
�
Adueling/train/gradients/dueling/eval_net/Advantage/add_grad/ShapeShape!dueling/eval_net/Advantage/MatMul*
T0*
out_type0*
_output_shapes
:
�
Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Qdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgsBroadcastGradientArgsAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ShapeCdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
?dueling/train/gradients/dueling/eval_net/Advantage/add_grad/SumSumdueling/train/gradients/AddNQdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeReshape?dueling/train/gradients/dueling/eval_net/Advantage/add_grad/SumAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
Adueling/train/gradients/dueling/eval_net/Advantage/add_grad/Sum_1Sumdueling/train/gradients/AddNSdueling/train/gradients/dueling/eval_net/Advantage/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
Edueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1ReshapeAdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Sum_1Cdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Ldueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_depsNoOpD^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeF^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1
�
Tdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependencyIdentityCdueling/train/gradients/dueling/eval_net/Advantage/add_grad/ReshapeM^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_deps*
T0*V
_classL
JHloc:@dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape*'
_output_shapes
:���������
�
Vdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency_1IdentityEdueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1M^dueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/group_deps*X
_classN
LJloc:@dueling/train/gradients/dueling/eval_net/Advantage/add_grad/Reshape_1*
_output_shapes

:*
T0
�
Edueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulMatMulTdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency"dueling/eval_net/Advantage/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
Gdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1MatMuldueling/eval_net/l1/ReluTdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
Odueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_depsNoOpF^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulH^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1
�
Wdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependencyIdentityEdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMulP^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Ydueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency_1IdentityGdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1P^dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*Z
_classP
NLloc:@dueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/MatMul_1
�
dueling/train/gradients/AddN_1AddNSdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependencyWdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency*
T0*T
_classJ
HFloc:@dueling/train/gradients/dueling/eval_net/Value/MatMul_grad/MatMul*
N*'
_output_shapes
:���������
�
>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradReluGraddueling/train/gradients/AddN_1dueling/eval_net/l1/Relu*
T0*'
_output_shapes
:���������
�
:dueling/train/gradients/dueling/eval_net/l1/add_grad/ShapeShapedueling/eval_net/l1/MatMul*
T0*
out_type0*
_output_shapes
:
�
<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
�
Jdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgsBroadcastGradientArgs:dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
8dueling/train/gradients/dueling/eval_net/l1/add_grad/SumSum>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradJdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
<dueling/train/gradients/dueling/eval_net/l1/add_grad/ReshapeReshape8dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum:dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
:dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum_1Sum>dueling/train/gradients/dueling/eval_net/l1/Relu_grad/ReluGradLdueling/train/gradients/dueling/eval_net/l1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
>dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1Reshape:dueling/train/gradients/dueling/eval_net/l1/add_grad/Sum_1<dueling/train/gradients/dueling/eval_net/l1/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes

:
�
Edueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_depsNoOp=^dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape?^dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1
�
Mdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependencyIdentity<dueling/train/gradients/dueling/eval_net/l1/add_grad/ReshapeF^dueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*O
_classE
CAloc:@dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape
�
Odueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency_1Identity>dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1F^dueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/group_deps*Q
_classG
ECloc:@dueling/train/gradients/dueling/eval_net/l1/add_grad/Reshape_1*
_output_shapes

:*
T0
�
>dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulMatMulMdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependencydueling/eval_net/l1/w1/read*
transpose_b(*
T0*'
_output_shapes
:���������*
transpose_a( 
�
@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1MatMul	dueling/sMdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0
�
Hdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_depsNoOp?^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulA^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1
�
Pdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependencyIdentity>dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMulI^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul*'
_output_shapes
:���������
�
Rdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependency_1Identity@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1I^dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@dueling/train/gradients/dueling/eval_net/l1/MatMul_grad/MatMul_1*
_output_shapes

:
�
=dueling/train/dueling/eval_net/l1/w1/RMSProp/Initializer/onesConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB*  �?*
dtype0*
_output_shapes

:
�
,dueling/train/dueling/eval_net/l1/w1/RMSProp
VariableV2*
shared_name *)
_class
loc:@dueling/eval_net/l1/w1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
3dueling/train/dueling/eval_net/l1/w1/RMSProp/AssignAssign,dueling/train/dueling/eval_net/l1/w1/RMSProp=dueling/train/dueling/eval_net/l1/w1/RMSProp/Initializer/ones*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
validate_shape(*
_output_shapes

:
�
1dueling/train/dueling/eval_net/l1/w1/RMSProp/readIdentity,dueling/train/dueling/eval_net/l1/w1/RMSProp*
_output_shapes

:*
T0*)
_class
loc:@dueling/eval_net/l1/w1
�
@dueling/train/dueling/eval_net/l1/w1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@dueling/eval_net/l1/w1*
valueB*    *
dtype0*
_output_shapes

:
�
.dueling/train/dueling/eval_net/l1/w1/RMSProp_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/w1
�
5dueling/train/dueling/eval_net/l1/w1/RMSProp_1/AssignAssign.dueling/train/dueling/eval_net/l1/w1/RMSProp_1@dueling/train/dueling/eval_net/l1/w1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
validate_shape(*
_output_shapes

:
�
3dueling/train/dueling/eval_net/l1/w1/RMSProp_1/readIdentity.dueling/train/dueling/eval_net/l1/w1/RMSProp_1*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:
�
=dueling/train/dueling/eval_net/l1/b1/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*  �?
�
,dueling/train/dueling/eval_net/l1/b1/RMSProp
VariableV2*
dtype0*
_output_shapes

:*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1*
	container *
shape
:
�
3dueling/train/dueling/eval_net/l1/b1/RMSProp/AssignAssign,dueling/train/dueling/eval_net/l1/b1/RMSProp=dueling/train/dueling/eval_net/l1/b1/RMSProp/Initializer/ones*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
1dueling/train/dueling/eval_net/l1/b1/RMSProp/readIdentity,dueling/train/dueling/eval_net/l1/b1/RMSProp*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
@dueling/train/dueling/eval_net/l1/b1/RMSProp_1/Initializer/zerosConst*)
_class
loc:@dueling/eval_net/l1/b1*
valueB*    *
dtype0*
_output_shapes

:
�
.dueling/train/dueling/eval_net/l1/b1/RMSProp_1
VariableV2*
shared_name *)
_class
loc:@dueling/eval_net/l1/b1*
	container *
shape
:*
dtype0*
_output_shapes

:
�
5dueling/train/dueling/eval_net/l1/b1/RMSProp_1/AssignAssign.dueling/train/dueling/eval_net/l1/b1/RMSProp_1@dueling/train/dueling/eval_net/l1/b1/RMSProp_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
validate_shape(*
_output_shapes

:
�
3dueling/train/dueling/eval_net/l1/b1/RMSProp_1/readIdentity.dueling/train/dueling/eval_net/l1/b1/RMSProp_1*
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
@dueling/train/dueling/eval_net/Value/w2/RMSProp/Initializer/onesConst*
_output_shapes

:*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB*  �?*
dtype0
�
/dueling/train/dueling/eval_net/Value/w2/RMSProp
VariableV2*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/w2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
6dueling/train/dueling/eval_net/Value/w2/RMSProp/AssignAssign/dueling/train/dueling/eval_net/Value/w2/RMSProp@dueling/train/dueling/eval_net/Value/w2/RMSProp/Initializer/ones*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:
�
4dueling/train/dueling/eval_net/Value/w2/RMSProp/readIdentity/dueling/train/dueling/eval_net/Value/w2/RMSProp*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:*
T0
�
Cdueling/train/dueling/eval_net/Value/w2/RMSProp_1/Initializer/zerosConst*
_output_shapes

:*,
_class"
 loc:@dueling/eval_net/Value/w2*
valueB*    *
dtype0
�
1dueling/train/dueling/eval_net/Value/w2/RMSProp_1
VariableV2*,
_class"
 loc:@dueling/eval_net/Value/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
8dueling/train/dueling/eval_net/Value/w2/RMSProp_1/AssignAssign1dueling/train/dueling/eval_net/Value/w2/RMSProp_1Cdueling/train/dueling/eval_net/Value/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
validate_shape(*
_output_shapes

:
�
6dueling/train/dueling/eval_net/Value/w2/RMSProp_1/readIdentity1dueling/train/dueling/eval_net/Value/w2/RMSProp_1*
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
@dueling/train/dueling/eval_net/Value/b2/RMSProp/Initializer/onesConst*
dtype0*
_output_shapes

:*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*  �?
�
/dueling/train/dueling/eval_net/Value/b2/RMSProp
VariableV2*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/b2*
	container *
shape
:*
dtype0
�
6dueling/train/dueling/eval_net/Value/b2/RMSProp/AssignAssign/dueling/train/dueling/eval_net/Value/b2/RMSProp@dueling/train/dueling/eval_net/Value/b2/RMSProp/Initializer/ones*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
4dueling/train/dueling/eval_net/Value/b2/RMSProp/readIdentity/dueling/train/dueling/eval_net/Value/b2/RMSProp*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
Cdueling/train/dueling/eval_net/Value/b2/RMSProp_1/Initializer/zerosConst*,
_class"
 loc:@dueling/eval_net/Value/b2*
valueB*    *
dtype0*
_output_shapes

:
�
1dueling/train/dueling/eval_net/Value/b2/RMSProp_1
VariableV2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name *,
_class"
 loc:@dueling/eval_net/Value/b2
�
8dueling/train/dueling/eval_net/Value/b2/RMSProp_1/AssignAssign1dueling/train/dueling/eval_net/Value/b2/RMSProp_1Cdueling/train/dueling/eval_net/Value/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
validate_shape(*
_output_shapes

:
�
6dueling/train/dueling/eval_net/Value/b2/RMSProp_1/readIdentity1dueling/train/dueling/eval_net/Value/b2/RMSProp_1*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:
�
Ddueling/train/dueling/eval_net/Advantage/w2/RMSProp/Initializer/onesConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB*  �?*
dtype0*
_output_shapes

:
�
3dueling/train/dueling/eval_net/Advantage/w2/RMSProp
VariableV2*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
:dueling/train/dueling/eval_net/Advantage/w2/RMSProp/AssignAssign3dueling/train/dueling/eval_net/Advantage/w2/RMSPropDdueling/train/dueling/eval_net/Advantage/w2/RMSProp/Initializer/ones*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
8dueling/train/dueling/eval_net/Advantage/w2/RMSProp/readIdentity3dueling/train/dueling/eval_net/Advantage/w2/RMSProp*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
Gdueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Initializer/zerosConst*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
valueB*    *
dtype0*
_output_shapes

:
�
5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
	container 
�
<dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/AssignAssign5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1Gdueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
:dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/readIdentity5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2*
_output_shapes

:
�
Ddueling/train/dueling/eval_net/Advantage/b2/RMSProp/Initializer/onesConst*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*  �?*
dtype0*
_output_shapes

:
�
3dueling/train/dueling/eval_net/Advantage/b2/RMSProp
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
	container 
�
:dueling/train/dueling/eval_net/Advantage/b2/RMSProp/AssignAssign3dueling/train/dueling/eval_net/Advantage/b2/RMSPropDdueling/train/dueling/eval_net/Advantage/b2/RMSProp/Initializer/ones*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
8dueling/train/dueling/eval_net/Advantage/b2/RMSProp/readIdentity3dueling/train/dueling/eval_net/Advantage/b2/RMSProp*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
_output_shapes

:
�
Gdueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Initializer/zerosConst*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
valueB*    *
dtype0*
_output_shapes

:
�
5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1
VariableV2*
shared_name *0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
	container *
shape
:*
dtype0*
_output_shapes

:
�
<dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/AssignAssign5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1Gdueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
:dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/readIdentity5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1*
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
_output_shapes

:
h
#dueling/train/RMSProp/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
`
dueling/train/RMSProp/decayConst*
valueB
 *fff?*
dtype0*
_output_shapes
: 
c
dueling/train/RMSProp/momentumConst*
valueB
 *    *
dtype0*
_output_shapes
: 
b
dueling/train/RMSProp/epsilonConst*
valueB
 *���.*
dtype0*
_output_shapes
: 
�
@dueling/train/RMSProp/update_dueling/eval_net/l1/w1/ApplyRMSPropApplyRMSPropdueling/eval_net/l1/w1,dueling/train/dueling/eval_net/l1/w1/RMSProp.dueling/train/dueling/eval_net/l1/w1/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonRdueling/train/gradients/dueling/eval_net/l1/MatMul_grad/tuple/control_dependency_1*
T0*)
_class
loc:@dueling/eval_net/l1/w1*
_output_shapes

:*
use_locking( 
�
@dueling/train/RMSProp/update_dueling/eval_net/l1/b1/ApplyRMSPropApplyRMSPropdueling/eval_net/l1/b1,dueling/train/dueling/eval_net/l1/b1/RMSProp.dueling/train/dueling/eval_net/l1/b1/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonOdueling/train/gradients/dueling/eval_net/l1/add_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@dueling/eval_net/l1/b1*
_output_shapes

:
�
Cdueling/train/RMSProp/update_dueling/eval_net/Value/w2/ApplyRMSPropApplyRMSPropdueling/eval_net/Value/w2/dueling/train/dueling/eval_net/Value/w2/RMSProp1dueling/train/dueling/eval_net/Value/w2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonUdueling/train/gradients/dueling/eval_net/Value/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*,
_class"
 loc:@dueling/eval_net/Value/w2*
_output_shapes

:
�
Cdueling/train/RMSProp/update_dueling/eval_net/Value/b2/ApplyRMSPropApplyRMSPropdueling/eval_net/Value/b2/dueling/train/dueling/eval_net/Value/b2/RMSProp1dueling/train/dueling/eval_net/Value/b2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonRdueling/train/gradients/dueling/eval_net/Value/add_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@dueling/eval_net/Value/b2*
_output_shapes

:*
use_locking( 
�
Gdueling/train/RMSProp/update_dueling/eval_net/Advantage/w2/ApplyRMSPropApplyRMSPropdueling/eval_net/Advantage/w23dueling/train/dueling/eval_net/Advantage/w2/RMSProp5dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonYdueling/train/gradients/dueling/eval_net/Advantage/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:*
use_locking( *
T0*0
_class&
$"loc:@dueling/eval_net/Advantage/w2
�
Gdueling/train/RMSProp/update_dueling/eval_net/Advantage/b2/ApplyRMSPropApplyRMSPropdueling/eval_net/Advantage/b23dueling/train/dueling/eval_net/Advantage/b2/RMSProp5dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1#dueling/train/RMSProp/learning_ratedueling/train/RMSProp/decaydueling/train/RMSProp/momentumdueling/train/RMSProp/epsilonVdueling/train/gradients/dueling/eval_net/Advantage/add_grad/tuple/control_dependency_1*0
_class&
$"loc:@dueling/eval_net/Advantage/b2*
_output_shapes

:*
use_locking( *
T0
�
dueling/train/RMSPropNoOpH^dueling/train/RMSProp/update_dueling/eval_net/Advantage/b2/ApplyRMSPropH^dueling/train/RMSProp/update_dueling/eval_net/Advantage/w2/ApplyRMSPropD^dueling/train/RMSProp/update_dueling/eval_net/Value/b2/ApplyRMSPropD^dueling/train/RMSProp/update_dueling/eval_net/Value/w2/ApplyRMSPropA^dueling/train/RMSProp/update_dueling/eval_net/l1/b1/ApplyRMSPropA^dueling/train/RMSProp/update_dueling/eval_net/l1/w1/ApplyRMSProp
m

dueling/s_Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
�
8dueling/target_net/l1/w1/Initializer/random_normal/shapeConst*+
_class!
loc:@dueling/target_net/l1/w1*
valueB"      *
dtype0*
_output_shapes
:
�
7dueling/target_net/l1/w1/Initializer/random_normal/meanConst*+
_class!
loc:@dueling/target_net/l1/w1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
9dueling/target_net/l1/w1/Initializer/random_normal/stddevConst*
_output_shapes
: *+
_class!
loc:@dueling/target_net/l1/w1*
valueB
 *���>*
dtype0
�
Gdueling/target_net/l1/w1/Initializer/random_normal/RandomStandardNormalRandomStandardNormal8dueling/target_net/l1/w1/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
seed2�
�
6dueling/target_net/l1/w1/Initializer/random_normal/mulMulGdueling/target_net/l1/w1/Initializer/random_normal/RandomStandardNormal9dueling/target_net/l1/w1/Initializer/random_normal/stddev*
_output_shapes

:*
T0*+
_class!
loc:@dueling/target_net/l1/w1
�
2dueling/target_net/l1/w1/Initializer/random_normalAdd6dueling/target_net/l1/w1/Initializer/random_normal/mul7dueling/target_net/l1/w1/Initializer/random_normal/mean*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
_output_shapes

:
�
dueling/target_net/l1/w1
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dueling/target_net/l1/w1*
	container 
�
dueling/target_net/l1/w1/AssignAssigndueling/target_net/l1/w12dueling/target_net/l1/w1/Initializer/random_normal*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/target_net/l1/w1/readIdentitydueling/target_net/l1/w1*+
_class!
loc:@dueling/target_net/l1/w1*
_output_shapes

:*
T0
�
*dueling/target_net/l1/b1/Initializer/ConstConst*+
_class!
loc:@dueling/target_net/l1/b1*
valueB*���=*
dtype0*
_output_shapes

:
�
dueling/target_net/l1/b1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *+
_class!
loc:@dueling/target_net/l1/b1*
	container *
shape
:
�
dueling/target_net/l1/b1/AssignAssigndueling/target_net/l1/b1*dueling/target_net/l1/b1/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
dueling/target_net/l1/b1/readIdentitydueling/target_net/l1/b1*
T0*+
_class!
loc:@dueling/target_net/l1/b1*
_output_shapes

:
�
dueling/target_net/l1/MatMulMatMul
dueling/s_dueling/target_net/l1/w1/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dueling/target_net/l1/addAdddueling/target_net/l1/MatMuldueling/target_net/l1/b1/read*
T0*'
_output_shapes
:���������
o
dueling/target_net/l1/ReluReludueling/target_net/l1/add*
T0*'
_output_shapes
:���������
�
;dueling/target_net/Value/w2/Initializer/random_normal/shapeConst*.
_class$
" loc:@dueling/target_net/Value/w2*
valueB"      *
dtype0*
_output_shapes
:
�
:dueling/target_net/Value/w2/Initializer/random_normal/meanConst*.
_class$
" loc:@dueling/target_net/Value/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
<dueling/target_net/Value/w2/Initializer/random_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@dueling/target_net/Value/w2*
valueB
 *���>*
dtype0
�
Jdueling/target_net/Value/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal;dueling/target_net/Value/w2/Initializer/random_normal/shape*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
seed2�*
dtype0*
_output_shapes

:*

seed
�
9dueling/target_net/Value/w2/Initializer/random_normal/mulMulJdueling/target_net/Value/w2/Initializer/random_normal/RandomStandardNormal<dueling/target_net/Value/w2/Initializer/random_normal/stddev*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
_output_shapes

:
�
5dueling/target_net/Value/w2/Initializer/random_normalAdd9dueling/target_net/Value/w2/Initializer/random_normal/mul:dueling/target_net/Value/w2/Initializer/random_normal/mean*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
_output_shapes

:
�
dueling/target_net/Value/w2
VariableV2*.
_class$
" loc:@dueling/target_net/Value/w2*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
"dueling/target_net/Value/w2/AssignAssigndueling/target_net/Value/w25dueling/target_net/Value/w2/Initializer/random_normal*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
validate_shape(*
_output_shapes

:*
use_locking(
�
 dueling/target_net/Value/w2/readIdentitydueling/target_net/Value/w2*
T0*.
_class$
" loc:@dueling/target_net/Value/w2*
_output_shapes

:
�
-dueling/target_net/Value/b2/Initializer/ConstConst*
dtype0*
_output_shapes

:*.
_class$
" loc:@dueling/target_net/Value/b2*
valueB*���=
�
dueling/target_net/Value/b2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *.
_class$
" loc:@dueling/target_net/Value/b2*
	container 
�
"dueling/target_net/Value/b2/AssignAssigndueling/target_net/Value/b2-dueling/target_net/Value/b2/Initializer/Const*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
validate_shape(*
_output_shapes

:
�
 dueling/target_net/Value/b2/readIdentitydueling/target_net/Value/b2*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
_output_shapes

:
�
dueling/target_net/Value/MatMulMatMuldueling/target_net/l1/Relu dueling/target_net/Value/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
dueling/target_net/Value/addAdddueling/target_net/Value/MatMul dueling/target_net/Value/b2/read*'
_output_shapes
:���������*
T0
�
?dueling/target_net/Advantage/w2/Initializer/random_normal/shapeConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB"      *
dtype0*
_output_shapes
:
�
>dueling/target_net/Advantage/w2/Initializer/random_normal/meanConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB
 *    *
dtype0*
_output_shapes
: 
�
@dueling/target_net/Advantage/w2/Initializer/random_normal/stddevConst*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
valueB
 *���>*
dtype0*
_output_shapes
: 
�
Ndueling/target_net/Advantage/w2/Initializer/random_normal/RandomStandardNormalRandomStandardNormal?dueling/target_net/Advantage/w2/Initializer/random_normal/shape*
dtype0*
_output_shapes

:*

seed*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
seed2�
�
=dueling/target_net/Advantage/w2/Initializer/random_normal/mulMulNdueling/target_net/Advantage/w2/Initializer/random_normal/RandomStandardNormal@dueling/target_net/Advantage/w2/Initializer/random_normal/stddev*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
_output_shapes

:
�
9dueling/target_net/Advantage/w2/Initializer/random_normalAdd=dueling/target_net/Advantage/w2/Initializer/random_normal/mul>dueling/target_net/Advantage/w2/Initializer/random_normal/mean*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
_output_shapes

:
�
dueling/target_net/Advantage/w2
VariableV2*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dueling/target_net/Advantage/w2*
	container *
shape
:
�
&dueling/target_net/Advantage/w2/AssignAssigndueling/target_net/Advantage/w29dueling/target_net/Advantage/w2/Initializer/random_normal*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
validate_shape(*
_output_shapes

:
�
$dueling/target_net/Advantage/w2/readIdentitydueling/target_net/Advantage/w2*
_output_shapes

:*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/w2
�
1dueling/target_net/Advantage/b2/Initializer/ConstConst*
_output_shapes

:*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
valueB*���=*
dtype0
�
dueling/target_net/Advantage/b2
VariableV2*
shape
:*
dtype0*
_output_shapes

:*
shared_name *2
_class(
&$loc:@dueling/target_net/Advantage/b2*
	container 
�
&dueling/target_net/Advantage/b2/AssignAssigndueling/target_net/Advantage/b21dueling/target_net/Advantage/b2/Initializer/Const*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/b2*
validate_shape(*
_output_shapes

:
�
$dueling/target_net/Advantage/b2/readIdentitydueling/target_net/Advantage/b2*
_output_shapes

:*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/b2
�
#dueling/target_net/Advantage/MatMulMatMuldueling/target_net/l1/Relu$dueling/target_net/Advantage/w2/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
 dueling/target_net/Advantage/addAdd#dueling/target_net/Advantage/MatMul$dueling/target_net/Advantage/b2/read*
T0*'
_output_shapes
:���������
m
+dueling/target_net/Q/Mean/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B :
�
dueling/target_net/Q/MeanMean dueling/target_net/Advantage/add+dueling/target_net/Q/Mean/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
dueling/target_net/Q/subSub dueling/target_net/Advantage/adddueling/target_net/Q/Mean*
T0*'
_output_shapes
:���������
�
dueling/target_net/Q/addAdddueling/target_net/Value/adddueling/target_net/Q/sub*'
_output_shapes
:���������*
T0
�
dueling/AssignAssignnatural/target_net/l1/w1natural/eval_net/l1/w1/read*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/w1*
validate_shape(*
_output_shapes

:
�
dueling/Assign_1Assignnatural/target_net/l1/b1natural/eval_net/l1/b1/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*+
_class!
loc:@natural/target_net/l1/b1
�
dueling/Assign_2Assignnatural/target_net/Q/w2natural/eval_net/Q/w2/read*
T0**
_class 
loc:@natural/target_net/Q/w2*
validate_shape(*
_output_shapes

:*
use_locking(
�
dueling/Assign_3Assignnatural/target_net/Q/b2natural/eval_net/Q/b2/read*
use_locking(*
T0**
_class 
loc:@natural/target_net/Q/b2*
validate_shape(*
_output_shapes

:
�
dueling/Assign_4Assigndueling/target_net/l1/w1dueling/eval_net/l1/w1/read*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/w1*
validate_shape(*
_output_shapes

:
�
dueling/Assign_5Assigndueling/target_net/l1/b1dueling/eval_net/l1/b1/read*
use_locking(*
T0*+
_class!
loc:@dueling/target_net/l1/b1*
validate_shape(*
_output_shapes

:
�
dueling/Assign_6Assigndueling/target_net/Value/w2dueling/eval_net/Value/w2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/w2
�
dueling/Assign_7Assigndueling/target_net/Value/b2dueling/eval_net/Value/b2/read*
use_locking(*
T0*.
_class$
" loc:@dueling/target_net/Value/b2*
validate_shape(*
_output_shapes

:
�
dueling/Assign_8Assigndueling/target_net/Advantage/w2"dueling/eval_net/Advantage/w2/read*2
_class(
&$loc:@dueling/target_net/Advantage/w2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
dueling/Assign_9Assigndueling/target_net/Advantage/b2"dueling/eval_net/Advantage/b2/read*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*2
_class(
&$loc:@dueling/target_net/Advantage/b2""�
trainable_variables��
�
natural/eval_net/l1/w1:0natural/eval_net/l1/w1/Assignnatural/eval_net/l1/w1/read:022natural/eval_net/l1/w1/Initializer/random_normal:08
�
natural/eval_net/l1/b1:0natural/eval_net/l1/b1/Assignnatural/eval_net/l1/b1/read:02*natural/eval_net/l1/b1/Initializer/Const:08
�
natural/eval_net/Q/w2:0natural/eval_net/Q/w2/Assignnatural/eval_net/Q/w2/read:021natural/eval_net/Q/w2/Initializer/random_normal:08
�
natural/eval_net/Q/b2:0natural/eval_net/Q/b2/Assignnatural/eval_net/Q/b2/read:02)natural/eval_net/Q/b2/Initializer/Const:08
�
natural/target_net/l1/w1:0natural/target_net/l1/w1/Assignnatural/target_net/l1/w1/read:024natural/target_net/l1/w1/Initializer/random_normal:08
�
natural/target_net/l1/b1:0natural/target_net/l1/b1/Assignnatural/target_net/l1/b1/read:02,natural/target_net/l1/b1/Initializer/Const:08
�
natural/target_net/Q/w2:0natural/target_net/Q/w2/Assignnatural/target_net/Q/w2/read:023natural/target_net/Q/w2/Initializer/random_normal:08
�
natural/target_net/Q/b2:0natural/target_net/Q/b2/Assignnatural/target_net/Q/b2/read:02+natural/target_net/Q/b2/Initializer/Const:08
�
dueling/eval_net/l1/w1:0dueling/eval_net/l1/w1/Assigndueling/eval_net/l1/w1/read:022dueling/eval_net/l1/w1/Initializer/random_normal:08
�
dueling/eval_net/l1/b1:0dueling/eval_net/l1/b1/Assigndueling/eval_net/l1/b1/read:02*dueling/eval_net/l1/b1/Initializer/Const:08
�
dueling/eval_net/Value/w2:0 dueling/eval_net/Value/w2/Assign dueling/eval_net/Value/w2/read:025dueling/eval_net/Value/w2/Initializer/random_normal:08
�
dueling/eval_net/Value/b2:0 dueling/eval_net/Value/b2/Assign dueling/eval_net/Value/b2/read:02-dueling/eval_net/Value/b2/Initializer/Const:08
�
dueling/eval_net/Advantage/w2:0$dueling/eval_net/Advantage/w2/Assign$dueling/eval_net/Advantage/w2/read:029dueling/eval_net/Advantage/w2/Initializer/random_normal:08
�
dueling/eval_net/Advantage/b2:0$dueling/eval_net/Advantage/b2/Assign$dueling/eval_net/Advantage/b2/read:021dueling/eval_net/Advantage/b2/Initializer/Const:08
�
dueling/target_net/l1/w1:0dueling/target_net/l1/w1/Assigndueling/target_net/l1/w1/read:024dueling/target_net/l1/w1/Initializer/random_normal:08
�
dueling/target_net/l1/b1:0dueling/target_net/l1/b1/Assigndueling/target_net/l1/b1/read:02,dueling/target_net/l1/b1/Initializer/Const:08
�
dueling/target_net/Value/w2:0"dueling/target_net/Value/w2/Assign"dueling/target_net/Value/w2/read:027dueling/target_net/Value/w2/Initializer/random_normal:08
�
dueling/target_net/Value/b2:0"dueling/target_net/Value/b2/Assign"dueling/target_net/Value/b2/read:02/dueling/target_net/Value/b2/Initializer/Const:08
�
!dueling/target_net/Advantage/w2:0&dueling/target_net/Advantage/w2/Assign&dueling/target_net/Advantage/w2/read:02;dueling/target_net/Advantage/w2/Initializer/random_normal:08
�
!dueling/target_net/Advantage/b2:0&dueling/target_net/Advantage/b2/Assign&dueling/target_net/Advantage/b2/read:023dueling/target_net/Advantage/b2/Initializer/Const:08"�
target_net_params�
�
natural/target_net/l1/w1:0
natural/target_net/l1/b1:0
natural/target_net/Q/w2:0
natural/target_net/Q/b2:0
dueling/target_net/l1/w1:0
dueling/target_net/l1/b1:0
dueling/target_net/Value/w2:0
dueling/target_net/Value/b2:0
!dueling/target_net/Advantage/w2:0
!dueling/target_net/Advantage/b2:0"<
train_op0
.
natural/train/RMSProp
dueling/train/RMSProp"�
eval_net_params�
�
natural/eval_net/l1/w1:0
natural/eval_net/l1/b1:0
natural/eval_net/Q/w2:0
natural/eval_net/Q/b2:0
dueling/eval_net/l1/w1:0
dueling/eval_net/l1/b1:0
dueling/eval_net/Value/w2:0
dueling/eval_net/Value/b2:0
dueling/eval_net/Advantage/w2:0
dueling/eval_net/Advantage/b2:0"�<
	variables�<�<
�
natural/eval_net/l1/w1:0natural/eval_net/l1/w1/Assignnatural/eval_net/l1/w1/read:022natural/eval_net/l1/w1/Initializer/random_normal:08
�
natural/eval_net/l1/b1:0natural/eval_net/l1/b1/Assignnatural/eval_net/l1/b1/read:02*natural/eval_net/l1/b1/Initializer/Const:08
�
natural/eval_net/Q/w2:0natural/eval_net/Q/w2/Assignnatural/eval_net/Q/w2/read:021natural/eval_net/Q/w2/Initializer/random_normal:08
�
natural/eval_net/Q/b2:0natural/eval_net/Q/b2/Assignnatural/eval_net/Q/b2/read:02)natural/eval_net/Q/b2/Initializer/Const:08
�
.natural/train/natural/eval_net/l1/w1/RMSProp:03natural/train/natural/eval_net/l1/w1/RMSProp/Assign3natural/train/natural/eval_net/l1/w1/RMSProp/read:02?natural/train/natural/eval_net/l1/w1/RMSProp/Initializer/ones:0
�
0natural/train/natural/eval_net/l1/w1/RMSProp_1:05natural/train/natural/eval_net/l1/w1/RMSProp_1/Assign5natural/train/natural/eval_net/l1/w1/RMSProp_1/read:02Bnatural/train/natural/eval_net/l1/w1/RMSProp_1/Initializer/zeros:0
�
.natural/train/natural/eval_net/l1/b1/RMSProp:03natural/train/natural/eval_net/l1/b1/RMSProp/Assign3natural/train/natural/eval_net/l1/b1/RMSProp/read:02?natural/train/natural/eval_net/l1/b1/RMSProp/Initializer/ones:0
�
0natural/train/natural/eval_net/l1/b1/RMSProp_1:05natural/train/natural/eval_net/l1/b1/RMSProp_1/Assign5natural/train/natural/eval_net/l1/b1/RMSProp_1/read:02Bnatural/train/natural/eval_net/l1/b1/RMSProp_1/Initializer/zeros:0
�
-natural/train/natural/eval_net/Q/w2/RMSProp:02natural/train/natural/eval_net/Q/w2/RMSProp/Assign2natural/train/natural/eval_net/Q/w2/RMSProp/read:02>natural/train/natural/eval_net/Q/w2/RMSProp/Initializer/ones:0
�
/natural/train/natural/eval_net/Q/w2/RMSProp_1:04natural/train/natural/eval_net/Q/w2/RMSProp_1/Assign4natural/train/natural/eval_net/Q/w2/RMSProp_1/read:02Anatural/train/natural/eval_net/Q/w2/RMSProp_1/Initializer/zeros:0
�
-natural/train/natural/eval_net/Q/b2/RMSProp:02natural/train/natural/eval_net/Q/b2/RMSProp/Assign2natural/train/natural/eval_net/Q/b2/RMSProp/read:02>natural/train/natural/eval_net/Q/b2/RMSProp/Initializer/ones:0
�
/natural/train/natural/eval_net/Q/b2/RMSProp_1:04natural/train/natural/eval_net/Q/b2/RMSProp_1/Assign4natural/train/natural/eval_net/Q/b2/RMSProp_1/read:02Anatural/train/natural/eval_net/Q/b2/RMSProp_1/Initializer/zeros:0
�
natural/target_net/l1/w1:0natural/target_net/l1/w1/Assignnatural/target_net/l1/w1/read:024natural/target_net/l1/w1/Initializer/random_normal:08
�
natural/target_net/l1/b1:0natural/target_net/l1/b1/Assignnatural/target_net/l1/b1/read:02,natural/target_net/l1/b1/Initializer/Const:08
�
natural/target_net/Q/w2:0natural/target_net/Q/w2/Assignnatural/target_net/Q/w2/read:023natural/target_net/Q/w2/Initializer/random_normal:08
�
natural/target_net/Q/b2:0natural/target_net/Q/b2/Assignnatural/target_net/Q/b2/read:02+natural/target_net/Q/b2/Initializer/Const:08
�
dueling/eval_net/l1/w1:0dueling/eval_net/l1/w1/Assigndueling/eval_net/l1/w1/read:022dueling/eval_net/l1/w1/Initializer/random_normal:08
�
dueling/eval_net/l1/b1:0dueling/eval_net/l1/b1/Assigndueling/eval_net/l1/b1/read:02*dueling/eval_net/l1/b1/Initializer/Const:08
�
dueling/eval_net/Value/w2:0 dueling/eval_net/Value/w2/Assign dueling/eval_net/Value/w2/read:025dueling/eval_net/Value/w2/Initializer/random_normal:08
�
dueling/eval_net/Value/b2:0 dueling/eval_net/Value/b2/Assign dueling/eval_net/Value/b2/read:02-dueling/eval_net/Value/b2/Initializer/Const:08
�
dueling/eval_net/Advantage/w2:0$dueling/eval_net/Advantage/w2/Assign$dueling/eval_net/Advantage/w2/read:029dueling/eval_net/Advantage/w2/Initializer/random_normal:08
�
dueling/eval_net/Advantage/b2:0$dueling/eval_net/Advantage/b2/Assign$dueling/eval_net/Advantage/b2/read:021dueling/eval_net/Advantage/b2/Initializer/Const:08
�
.dueling/train/dueling/eval_net/l1/w1/RMSProp:03dueling/train/dueling/eval_net/l1/w1/RMSProp/Assign3dueling/train/dueling/eval_net/l1/w1/RMSProp/read:02?dueling/train/dueling/eval_net/l1/w1/RMSProp/Initializer/ones:0
�
0dueling/train/dueling/eval_net/l1/w1/RMSProp_1:05dueling/train/dueling/eval_net/l1/w1/RMSProp_1/Assign5dueling/train/dueling/eval_net/l1/w1/RMSProp_1/read:02Bdueling/train/dueling/eval_net/l1/w1/RMSProp_1/Initializer/zeros:0
�
.dueling/train/dueling/eval_net/l1/b1/RMSProp:03dueling/train/dueling/eval_net/l1/b1/RMSProp/Assign3dueling/train/dueling/eval_net/l1/b1/RMSProp/read:02?dueling/train/dueling/eval_net/l1/b1/RMSProp/Initializer/ones:0
�
0dueling/train/dueling/eval_net/l1/b1/RMSProp_1:05dueling/train/dueling/eval_net/l1/b1/RMSProp_1/Assign5dueling/train/dueling/eval_net/l1/b1/RMSProp_1/read:02Bdueling/train/dueling/eval_net/l1/b1/RMSProp_1/Initializer/zeros:0
�
1dueling/train/dueling/eval_net/Value/w2/RMSProp:06dueling/train/dueling/eval_net/Value/w2/RMSProp/Assign6dueling/train/dueling/eval_net/Value/w2/RMSProp/read:02Bdueling/train/dueling/eval_net/Value/w2/RMSProp/Initializer/ones:0
�
3dueling/train/dueling/eval_net/Value/w2/RMSProp_1:08dueling/train/dueling/eval_net/Value/w2/RMSProp_1/Assign8dueling/train/dueling/eval_net/Value/w2/RMSProp_1/read:02Edueling/train/dueling/eval_net/Value/w2/RMSProp_1/Initializer/zeros:0
�
1dueling/train/dueling/eval_net/Value/b2/RMSProp:06dueling/train/dueling/eval_net/Value/b2/RMSProp/Assign6dueling/train/dueling/eval_net/Value/b2/RMSProp/read:02Bdueling/train/dueling/eval_net/Value/b2/RMSProp/Initializer/ones:0
�
3dueling/train/dueling/eval_net/Value/b2/RMSProp_1:08dueling/train/dueling/eval_net/Value/b2/RMSProp_1/Assign8dueling/train/dueling/eval_net/Value/b2/RMSProp_1/read:02Edueling/train/dueling/eval_net/Value/b2/RMSProp_1/Initializer/zeros:0
�
5dueling/train/dueling/eval_net/Advantage/w2/RMSProp:0:dueling/train/dueling/eval_net/Advantage/w2/RMSProp/Assign:dueling/train/dueling/eval_net/Advantage/w2/RMSProp/read:02Fdueling/train/dueling/eval_net/Advantage/w2/RMSProp/Initializer/ones:0
�
7dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1:0<dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Assign<dueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/read:02Idueling/train/dueling/eval_net/Advantage/w2/RMSProp_1/Initializer/zeros:0
�
5dueling/train/dueling/eval_net/Advantage/b2/RMSProp:0:dueling/train/dueling/eval_net/Advantage/b2/RMSProp/Assign:dueling/train/dueling/eval_net/Advantage/b2/RMSProp/read:02Fdueling/train/dueling/eval_net/Advantage/b2/RMSProp/Initializer/ones:0
�
7dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1:0<dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Assign<dueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/read:02Idueling/train/dueling/eval_net/Advantage/b2/RMSProp_1/Initializer/zeros:0
�
dueling/target_net/l1/w1:0dueling/target_net/l1/w1/Assigndueling/target_net/l1/w1/read:024dueling/target_net/l1/w1/Initializer/random_normal:08
�
dueling/target_net/l1/b1:0dueling/target_net/l1/b1/Assigndueling/target_net/l1/b1/read:02,dueling/target_net/l1/b1/Initializer/Const:08
�
dueling/target_net/Value/w2:0"dueling/target_net/Value/w2/Assign"dueling/target_net/Value/w2/read:027dueling/target_net/Value/w2/Initializer/random_normal:08
�
dueling/target_net/Value/b2:0"dueling/target_net/Value/b2/Assign"dueling/target_net/Value/b2/read:02/dueling/target_net/Value/b2/Initializer/Const:08
�
!dueling/target_net/Advantage/w2:0&dueling/target_net/Advantage/w2/Assign&dueling/target_net/Advantage/w2/read:02;dueling/target_net/Advantage/w2/Initializer/random_normal:08
�
!dueling/target_net/Advantage/b2:0&dueling/target_net/Advantage/b2/Assign&dueling/target_net/Advantage/b2/read:023dueling/target_net/Advantage/b2/Initializer/Const:08B�