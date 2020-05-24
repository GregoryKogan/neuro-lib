class Matrix{
  constructor(Rows, Coloumns){
    this.Rows = Rows;
    this.Coloumns = Coloumns;

    this.Values = [];
    for (let i = 0; i < this.Rows; ++i){
      this.Values[i] = [];
      for (let j = 0; j < this.Coloumns; ++j){
        this.Values[i][j] = 0;
      }
    }
  }

  static Multiply(m1, m2){
    if (m1.Coloumns != m2.Rows) return undefined;
    let Result = new Matrix(m1.Rows, m2.Coloumns);
    for (let i = 0; i < Result.Rows; ++i){
      for (let j = 0; j < Result.Coloumns; ++j){
        let Sum = 0;
        for (let k = 0; k < m1.Coloumns; ++k){
          Sum += m1.Values[i][k] * m2.Values[k][j];
        }
        Result.Values[i][j] = Sum;
      }
    }
    return Result;
  }

  Multiply(n){
    if (n instanceof Matrix){
      for (let i = 0; i < this.Rows; ++i){
        for (let j = 0; j < this.Coloumns; ++j){
          this.Values[i][j] *= n.Values[i][j];
        }
      }
    }
    else{
      for (let i = 0; i < this.Rows; ++i){
        for (let j = 0; j < this.Coloumns; ++j){
          this.Values[i][j] *= n;
        }
      }
    }
  }

  FillRandom(){
    for (let i = 0; i < this.Rows; ++i){
      for (let j = 0; j < this.Coloumns; ++j){
        this.Values[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  static Transpose(m1){
    let Result = new Matrix(m1.Coloumns, m1.Rows);
    for (let i = 0; i < m1.Rows; ++i){
      for (let j = 0; j < m1.Coloumns; ++j){
        Result.Values[j][i] = m1.Values[i][j];
      }
    }
    return Result;
  }

  Add(n) {
    if (n instanceof Matrix) {
      for (let i = 0; i < this.Rows; i++) {
        for (let j = 0; j < this.Coloumns; j++) {
          this.Values[i][j] += n.Values[i][j];
        }
      }
    }
    else {
      for (let i = 0; i < this.Rows; i++) {
        for (let j = 0; j < this.Coloumns; j++) {
          this.Values[i][j] += n;
        }
      }
    }
  }

  static Substract(m1, m2){
    let Result = new Matrix(m1.Rows, m1.Coloumns);
    for (let i = 0; i < Result.Rows; ++i){
      for (let j = 0; j < Result.Coloumns; ++j){
        Result.Values[i][j] = m1.Values[i][j] - m2.Values[i][j];
      }
    }
    return Result;
  }

  static ApplyFunction(M, F){
    let Result = new Matrix(M.Rows, M.Coloumns);
    for (let i = 0; i < M.Rows; ++i){
      for (let j = 0; j < M.Coloumns; ++j){
        let Value = M.Values[i][j];
        Result.Values[i][j] = F(Value);
      }
    }
    return Result;
  }

  ApplyFunction(F){
    for (let i = 0; i < this.Rows; ++i){
      for (let j = 0; j < this.Coloumns; ++j){
        let Value = this.Values[i][j];
        this.Values[i][j] = F(Value);
      }
    }
  }

  static MakeArray(InputMatrix){
    let A = [];
    for (let i = 0; i < InputMatrix.Rows; ++i){
      for (let j = 0; j < InputMatrix.Coloumns; ++j){
        A.push(InputMatrix.Values[i][j]);
      }
    }
    return A;
  }

  static MakeMatrix(InputArray){
    let M = new Matrix(InputArray.length, 1);
    for (let i = 0; i < InputArray.length; ++i){
      M.Values[i][0] = InputArray[i];
    }
    return M;
  }

  Print(){
    console.table(this.Values);
  }

}

function Sigmoid(x){
  return 1 / (1 + Math.exp(-x));
}

function SigmoidDerivative(S){
  //return Sigmoid(x) * (1 - Sigmoid(x));
  return S * (1 - S);
}

class NeuralNetwork{
  constructor(InputNum, HiddenNum, OutputNum){
    this.NumOfInputNeurons = InputNum;
    this.NumOfHiddenNeurons = HiddenNum;
    this.NumOfOutputNeurons = OutputNum;

    this.WeightsIH = new Matrix(this.NumOfHiddenNeurons, this.NumOfInputNeurons);
    this.WeightsHO = new Matrix(this.NumOfOutputNeurons, this.NumOfHiddenNeurons);
    this.WeightsIH.FillRandom();
    this.WeightsHO.FillRandom();

    this.BiasH = new Matrix(this.NumOfHiddenNeurons, 1);
    this.BiasO = new Matrix(this.NumOfOutputNeurons, 1);
    this.BiasH.FillRandom();
    this.BiasO.FillRandom();

    this.LearningRate = 0.1;
  }

  Predict(InputArray){
    let Input = Matrix.MakeMatrix(InputArray);

    let Hidden = Matrix.Multiply(this.WeightsIH, Input);
    Hidden.Add(this.BiasH);
    Hidden.ApplyFunction(Sigmoid);

    let Output = Matrix.Multiply(this.WeightsHO, Hidden);
    Output.Add(this.BiasO);
    Output.ApplyFunction(Sigmoid);

    return Matrix.MakeArray(Output);
  }

  Train(InputArray, Answer){
    let Input = Matrix.MakeMatrix(InputArray);

    let Hidden = Matrix.Multiply(this.WeightsIH, Input);
    Hidden.Add(this.BiasH);
    Hidden.ApplyFunction(Sigmoid);

    let Output = Matrix.Multiply(this.WeightsHO, Hidden);
    Output.Add(this.BiasO);
    Output.ApplyFunction(Sigmoid);

    let Target = Matrix.MakeMatrix(Answer);

    let OutputError = Matrix.Substract(Target, Output);

    let Gradient = Matrix.ApplyFunction(Output, SigmoidDerivative);
    Gradient.Multiply(OutputError);
    Gradient.Multiply(this.LearningRate);

    let TransposedHidden = Matrix.Transpose(Hidden);
    let WeightDeltaHO = Matrix.Multiply(Gradient, TransposedHidden);

    this.WeightsHO.Add(WeightDeltaHO);
    this.BiasO.Add(Gradient);

    let HiddenError = Matrix.Multiply(Matrix.Transpose(this.WeightsHO), OutputError);

    let HiddenGradient = Matrix.ApplyFunction(Hidden, SigmoidDerivative);
    HiddenGradient.Multiply(HiddenError);
    HiddenGradient.Multiply(this.LearningRate);

    let TransposedInput = Matrix.Transpose(Input);
    let WeightDeltaIH = Matrix.Multiply(HiddenGradient, TransposedInput);

    this.WeightsIH.Add(WeightDeltaIH);
    this.BiasH.Add(HiddenGradient);
  }

  SetLearningRate(n){
    this.LearningRate = n;
  }
}
