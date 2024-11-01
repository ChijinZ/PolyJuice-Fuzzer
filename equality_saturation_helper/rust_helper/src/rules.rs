use crate::model::{ComputExprIR, TensorInfoForAnalysis};
use egg::{rewrite as rw, *};
use log::info;

fn is_shape_equal<A>(
    var1: &'static str,
    var2: &'static str,
) -> impl Fn(&mut EGraph<ComputExprIR, A>, Id, &Subst) -> bool
where
    A: Analysis<ComputExprIR>,
    A::Data: PartialEq,
{
    let var1 = var1.parse().unwrap();
    let var2 = var2.parse().unwrap();
    move |egraph, id, subst| {
        let var1_eclass_id = subst[var1];
        let var2_eclass_id = subst[var2];
        info!("var1_eclass_id: {:?}", var1_eclass_id);
        info!("var2_eclass_id: {:?}", var2_eclass_id);
        info!("passed id: {:?}", id);

        egraph[var1_eclass_id].data == egraph[var2_eclass_id].data
    }
}

fn is_dim_larger_than_a_value<A>(
    var: &'static str,
    value: usize,
) -> impl Fn(&mut EGraph<ComputExprIR, A>, Id, &Subst) -> bool
where
    A: Analysis<ComputExprIR>,
    A::Data: TensorInfoForAnalysis,
{
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        let var_eclass_id = subst[var];
        let var_shape = egraph[var_eclass_id].data.shape();
        var_shape.len() > value
    }
}

pub fn rules<A>() -> Vec<Rewrite<ComputExprIR, A>>
where
    A: Analysis<ComputExprIR> + 'static,
    A::Data: PartialEq + TensorInfoForAnalysis,
{
    vec![
        rw!("ewadd-is-associative-0"            ; "(ewadd ?x (ewadd ?y ?z))"                                            => "(ewadd (ewadd ?x ?y) ?z)"),
        rw!("ewadd-is-associative-1"            ; "(ewadd (ewadd ?x ?y) ?z)"                                            => "(ewadd ?x (ewadd ?y ?z))"),
        rw!("ewadd-is-commutative"              ; "(ewadd ?x ?y) "                                                      => "(ewadd ?y ?x)"),
        rw!("ewmul-is-associative-0"            ; "(ewmul ?x (ewmul ?y ?z))"                                            => "(ewmul (ewmul ?x ?y) ?z)"),
        rw!("ewmul-is-associative-1"            ; "(ewmul (ewmul ?x ?y) ?z)"                                            => "(ewmul ?x (ewmul ?y ?z))"),
        rw!("ewmul-is-commutative"              ; "(ewmul ?x ?y) "                                                      => "(ewmul ?y ?x)"),
        rw!("ewmul-distributivity-0"            ; "(ewmul (ewadd ?x ?y) ?z)"                                            => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
        rw!("ewmul-distributivity-1"            ; "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"                                 => "(ewmul (ewadd ?x ?y) ?z)"),
        rw!("matmul-is-associative-0"           ; "(matmul ?x (matmul ?y ?z))"                                          => "(matmul (matmul ?x ?y) ?z)"),
        rw!("matmul-is-linear-0"                ; "(ewadd (matmul ?x ?y) (matmul ?x ?z))"                               => "(matmul ?x (ewadd ?y ?z))"),
        rw!("matmul-is-linear-1"                ; "(matmul ?x (ewadd ?y ?z))"                                           => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("trans-reduce";
            "(transpose (transpose ?x ?a ?b) ?a ?b)" => "?x"
        ),
        rw!("add-trans";
            "(ewadd ?x ?y)" => "(transpose (ewadd (transpose ?x 0 1) (transpose ?y 0 1)) 0 1)"
            if is_shape_equal("?x", "?y")
            if is_dim_larger_than_a_value("?x", 1)
            if is_dim_larger_than_a_value("?y", 1)
        ),
        rw!("mul-trans";
            "(ewmul ?x ?y)" => "(transpose (ewmul (transpose ?x 0 1) (transpose ?y 0 1)) 0 1)"
            if is_shape_equal("?x", "?y")
            if is_dim_larger_than_a_value("?x", 1)
            if is_dim_larger_than_a_value("?y", 1)
        ),
        rw!("matmul-trans";
            "(matmul ?x ?y)" => "(transpose (matmul (transpose ?x 0 1) (transpose ?y 0 1)) 0 1)"
            if is_dim_larger_than_a_value("?x", 1)
            if is_dim_larger_than_a_value("?y", 1)
        ),
        rw!("trans-add";
            "(transpose (ewadd ?x ?y) ?a ?b)" => "(ewadd (transpose ?x ?a ?b)  (transpose ?y ?a ?b))"
            if is_shape_equal("?x", "?y")
        ),
        rw!("trans-mul";
            "(transpose (ewmul ?x ?y) ?a ?b)" => "(ewmul (transpose ?x ?a ?b)  (transpose ?y ?a ?b))"
            if is_shape_equal("?x", "?y")
        ),
        rw!("concat-split-0";
            "(ewadd ?x ?y)" => "(split2_0 (concat (ewadd ?x ?y) (ewadd ?x ?y) 0) 0)"
            if is_dim_larger_than_a_value("?x", 0)
            if is_dim_larger_than_a_value("?y", 0)
        ),
        rw!("concat-split-1";
            "(ewmul ?x ?y)" => "(split2_0 (concat (ewmul ?x ?y) (ewmul ?x ?y) 0) 0)"
            if is_dim_larger_than_a_value("?x", 0)
            if is_dim_larger_than_a_value("?y", 0)
        ),
        rw!("concat-split-2"; "(transpose ?x ?a ?b)" => "(split2_0 (concat (transpose ?x ?a ?b) (transpose ?x ?a ?b) ?a) ?a)"),
        rw!("concat-split-3"; "(concat ?x ?y ?a)" => "(split2_0 (concat (concat ?x ?y ?a) (concat ?x ?y ?a) ?a) ?a)"),
        rw!("geometry-of-concatenation"; "(concat (concat ?x ?y ?a) (concat ?z ?w ?a) ?b)" => "(concat (concat ?x ?z ?b) (concat ?y ?w ?b) ?a)"),
    ]
}

#[allow(unused)]
pub fn rules2<A: Analysis<ComputExprIR>>() -> Vec<Rewrite<ComputExprIR, A>> {
    vec![
        rw!("ewadd-is-associative-0"            ; "(ewadd ?x (ewadd ?y ?z))"                                            => "(ewadd (ewadd ?x ?y) ?z)"),
        rw!("ewadd-is-associative-1"            ; "(ewadd (ewadd ?x ?y) ?z)"                                            => "(ewadd ?x (ewadd ?y ?z))"),
        rw!("ewadd-is-commutative"              ; "(ewadd ?x ?y) "                                                      => "(ewadd ?y ?x)"),
        rw!("ewmul-is-associative-0"            ; "(ewmul ?x (ewmul ?y ?z))"                                            => "(ewmul (ewmul ?x ?y) ?z)"),
        rw!("ewmul-is-associative-1"            ; "(ewmul (ewmul ?x ?y) ?z)"                                            => "(ewmul ?x (ewmul ?y ?z))"),
        rw!("ewmul-is-commutative"              ; "(ewmul ?x ?y) "                                                      => "(ewmul ?y ?x)"),
        rw!("ewmul-distributivity-0"            ; "(ewmul (ewadd ?x ?y) ?z)"                                            => "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"),
        rw!("ewmul-distributivity-1"            ; "(ewadd (ewmul ?x ?z) (ewmul ?y ?z))"                                 => "(ewmul (ewadd ?x ?y) ?z)"),
        rw!("matmul-is-associative-0"           ; "(matmul ?x (matmul ?y ?z))"                                          => "(matmul (matmul ?x ?y) ?z)"),
        // rw!("matmul-is-associative-1"           ; "(matmul ?x (matmul ?y ?z))"                                          => "(matmul (matmul ?x ?y) ?z)"),
        rw!("matmul-is-linear-0"                ; "(ewadd (matmul ?x ?y) (matmul ?x ?z))"                               => "(matmul ?x (ewadd ?y ?z))"),
        rw!("matmul-is-linear-1"                ; "(matmul ?x (ewadd ?y ?z))"                                           => "(ewadd (matmul ?x ?y) (matmul ?x ?z))"),
        rw!("matmul-and-transpose-0"            ; "(transpose (matmul ?x ?y) ?a ?b)"                                    => "(matmul (transpose ?y ?a ?b) (transpose ?x ?a ?b))"),
        rw!("matmul-and-transpose-1"            ; "(matmul (transpose ?y ?a ?b) (transpose ?x ?a ?b))"                  => "(transpose (matmul ?x ?y) ?a ?b)"),
        rw!("transpose-is-its-own-0"            ; "(transpose (transpose ?x ?a ?b) ?a ?b)"                              => "?x"),
        // rw!("transpose-is-its-own-1"            ; "(ewadd ?x ?y)"                                                       => "(transpose (transpose (ewadd ?x ?y) 1 0) 0 1)"), // we should ensure the two dimensions are equal
        // rw!("transpose-is-its-own-2"            ; "(ewmul ?x ?y)"                                                       => "(transpose (transpose (ewmul ?x ?y) 1 0) 0 1)"),
        rw!("transpose-is-its-own-3"            ; "(concat ?x ?y ?a)"                                                   => "(transpose (transpose (concat ?x ?y ?a) 1 0) 0 1)"),
        // rw!("transpose-is-its-own-4"            ; "(tensor ?x ?a ?b)"                                                   => "(transpose (transpose (tensor ?x ?a ?b) 1 0) 0 1)"),
        rw!("operator-commutativity-0"          ; "(transpose (ewadd ?x ?y) ?a ?b)"                                     => "(ewadd (transpose ?x ?a ?b)  (transpose ?y ?a ?b))"),
        rw!("operator-commutativity-1"          ; "(transpose (ewmul ?x ?y) ?a ?b)"                                     => "(ewmul (transpose ?x ?a ?b)  (transpose ?y ?a ?b))"),
        rw!("concat-split-0"                    ; "(ewadd ?x ?y)"                                                       => "(split2_0 (concat (ewadd ?x ?y) (ewadd ?x ?y) 0) 0)"),
        rw!("concat-split-1"                    ; "(ewmul ?x ?y)"                                                       => "(split2_0 (concat (ewmul ?x ?y) (ewmul ?x ?y) 0) 0)"),
        rw!("concat-split-2"                    ; "(transpose ?x ?a ?b)"                                                => "(split2_0 (concat (transpose ?x ?a ?b) (transpose ?x ?a ?b) ?a) ?a)"),
        rw!("concat-split-3"                    ; "(concat ?x ?y ?a)"                                                   => "(split2_0 (concat (concat ?x ?y ?a) (concat ?x ?y ?a) ?a) ?a)"),
        // rw!("concat-split-4"                    ; "(matmul ?x ?y)"                                                      => "(split2_0 (concat (matmul ?x ?y) (matmul ?x ?y) 0) 0)"),
        // rw!("concat-split-5"                    ; "(tensor ?x ?a ?b)"                                                   => "(split2_0 (concat (tensor ?x ?a ?b) (tensor ?x ?a ?b) 0) 0)"),
        // rw!("split-concat-0"                    ; "(ewadd ?x ?y)"                                                       => "(concat (split2_0 (ewadd ?x ?y) 0) (split2_1 (ewadd ?x ?y) 0) 0)"),
        // rw!("split-concat-1"                    ; "(ewmul ?x ?y)"                                                       => "(concat (split2_0 (ewmul ?x ?y) 0) (split2_1 (ewmul ?x ?y) 0) 0)"),
        // rw!("split-concat-2"                    ; "(transpose ?x ?a ?b)"                                                => "(concat (split2_0 (transpose ?x ?a ?b) 0) (split2_1 (transpose ?x ?a ?b) 0) 0)"),
        // rw!("split-concat-3"                    ; "(concat ?x ?y ?a)"                                                   => "(concat (split2_0 (concat ?x ?y ?a) 0) (split2_1 (concat ?x ?y ?a) 0) 0)"),
        // rw!("split-concat-4"                    ; "(matmul ?x ?y)"                                                      => "(concat (split2_0 (matmul ?x ?y) 0) (split2_1 (matmul ?x ?y) 0) 0)"),
        // rw!("split-concat-5"                    ; "(tensor ?x ?a ?b)"                                                   => "(concat (split2_0 (tensor ?x ?a ?b) 0) (split2_1 (tensor ?x ?a ?b) 0) 0)"),
        //
        rw!("geometry-of-concatenation"       ; "(concat (concat ?x ?y ?a) (concat ?z ?w ?a) ?b)"                       => "(concat (concat ?x ?z ?b) (concat ?y ?w ?b) ?a)"),
        // rw!("operator-commutativity-6"        ; "(concat (ewadd ?x ?y) (ewadd ?z ?w) ?a)"                               => "(ewadd (concat ?x ?z ?a) (concat ?y ?w ?a))"),
        // rw!("operator-commutativity-7"        ; "(concat (ewmul ?x ?y) (ewmul ?z ?w) ?a)"                               => "(ewmul (concat ?x ?z ?a) (concat ?y ?w ?a))"),
        // rw!("operator-commutativity-8"        ; "(concat (relu ?x) (relu ?y) ?a)"                                       => "(relu (concat ?x ?y ?a))"),

        // rw!("concatenation-and-transpose-0"   ; "(concat ?x ?y ?a)"                                                     => "(transpose (concat (transpose ?x ?a 0) (transpose ?y ?a 0) 0) ?a 0)"),
        // rw!("concatenation-and-transpose-1"   ; "(concat (transpose ?x ?a ?b) (transpose ?y ?a ?b) ?b)"                 => "(transpose (concat ?x ?y ?a) ?a ?b)"),
    ]
}
