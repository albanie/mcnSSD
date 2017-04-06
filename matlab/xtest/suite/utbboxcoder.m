classdef utbboxcoder < matlab.unittest.TestCase
  methods (Test)

    function basic(test)
      % sample bounding boxes in the 
      % [xmin ymin width height] format
      minXY = rand(10,2) ;
      remainXY = ones(size(minXY)) - minXY ;
      WH = rand(size(minXY)) .* remainXY ;

      % compute alternative formats
      minMax = [ minXY minXY + WH ] ;
      minWH = [ minXY WH ] ;
      cenWH = [ minXY + WH / 2 WH ] ;

      % test conversions
      test.verifyEqual(minMax, bboxCoder(minWH, 'MinWH', 'MinMax'), 'AbsTol', 1e-10) ;
      test.verifyEqual(minMax, bboxCoder(cenWH, 'CenWH', 'MinMax'), 'AbsTol', 1e-10) ;

      test.verifyEqual(minWH, bboxCoder(minMax, 'MinMax', 'MinWH'), 'AbsTol', 1e-10) ;
      test.verifyEqual(minWH, bboxCoder(cenWH, 'CenWH', 'MinWH'), 'AbsTol', 1e-10) ;

      test.verifyEqual(cenWH, bboxCoder(minMax, 'MinMax', 'CenWH'), 'AbsTol', 1e-10) ;
      test.verifyEqual(cenWH, bboxCoder(minWH, 'MinWH', 'CenWH'), 'AbsTol', 1e-10) ;
    end

  end
end
