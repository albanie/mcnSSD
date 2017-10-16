classdef nnmultiboxloss < nntest
  methods (Test)

    function basic(test)
      x = test.randn(1) ;
      l = test.randn(1) ;
      y = vl_nnmultiboxloss(x, l) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      [dzdx, dzdl] = vl_nnmultiboxloss(x, l, dzdy) ;
      test.der(@(x) vl_nnmultiboxloss(x, l), x, dzdy, dzdx, 1e-3*test.range) ;
      test.der(@(l) vl_nnmultiboxloss(x, l), l, dzdy, dzdl, 1e-3*test.range) ;
    end
  end
end
