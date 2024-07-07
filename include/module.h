// module.h
#ifndef MODULE_H
#define MODULE_H

#include "utils.h"
#include <vector>
class OptimizerAdam {
public:
    explicit OptimizerAdam(float _lr = 0.01, float _beta_1 = 0.9, float _beta_2 = 0.999, float _eps = 1e-8);

public:
    float learning_rate;
    float beta_1;
    float beta_2;
    float eps;

    float m_bias;  // 偏置项的一阶矩估计
    float v_bias;  // 偏置项的二阶矩估计
    float m_weights;  // 权重的一阶矩估计
    float v_weights;  // 权重的二阶矩估计
    int t;  // 时间步

    // gpu 
    float *beta_1_gpu;
    float *beta_2_gpu;
    float *eps_gpu;
    float *m_bias_gpu;
    float *v_bias_gpu;
    float *m_weights_gpu;
    float *v_weights_gpu;
    int t_gpu;
    void to_gpu(void);
};

class Module {
public:
    int inputshape = 0;
    int outputshape = 0;

    // for gpu 
    float *inp, *out;
    virtual void forward_gpu(float *_inp, float *_out,int batch_size) =0 ;
    virtual void backward_gpu(float* gradInput,float* gradOutput,int batch_size) = 0;
    virtual void update_gpu(float* gradInput,OptimizerAdam& optim,int batch_size) = 0 ;
    //

    Module(int inshape,int outshape): inputshape(inshape),outputshape(outshape){}
    virtual Vector forward(const Vector& input) const = 0;
    virtual Matrix forward_batch( Matrix& input)  = 0;
    virtual Matrix backward_batch(const Matrix& gradInput)  = 0;
    virtual void update(Matrix& gradInput,OptimizerAdam& optim) =0 ;

    virtual int getInputSize() const = 0;

    virtual int getOutputSize() const = 0;
    virtual ~Module(){
        cudaFree(inp);
        cudaFree(out);
    }
};

class DenseLayer : public Module {
public:
    Matrix weights;
    Matrix copy_weights;
    Vector bias;
    Matrix saved_input;
    
    //For gpu part -----------------------------
    float *weights_gpu;
    float *bias_gpu;
    float *cp_weights;
    int bs, n_in, n_out, sz_weights, n_block_rows, n_block_cols;
    //------------------------------------------

public:
    DenseLayer(const Matrix& w, const Vector& b,int inshape,int outshape);

    Vector forward(const Vector& input) const override;
    Vector backward(const Vector& dLdY);
    Matrix forward_batch( Matrix& input)  override;
    Vector backward(const Vector& dLdY) const;
    Matrix backward_batch(const Matrix& gradInput)  override;
    void update(Matrix& gradInput,OptimizerAdam& optim) override;

    //-----------------------------
    void forward_gpu(float *_inp, float *_out,int batch_size) override;
    void backward_gpu(float* gradInput,float* gradOutput,int batch_size) override;
    void update_gpu(float* gradInput,OptimizerAdam& optim,int batch_size) override ;
    //------------------------
    int getInputSize() const override;

    int getOutputSize() const override;
};

class SinLayer : public Module {
public:
    const float w0 = 30.0;
    Matrix saved_input;

    //------------------------gpu-------
    int n_blocks;
    //-----------------------------------
    
public:
    SinLayer(int inshape,int outshape);
    Vector forward(const Vector& input) const override;
    Matrix forward_batch( Matrix& input)  override;
    Vector backward(const Vector& dL_dy, Vector& input);
    Matrix backward_batch(const Matrix& gradInput)  override;
    void update(Matrix& gradInput,OptimizerAdam& optim) override;
    void backward_gpu(float* gradInput,float* gradOutput,int batch_size) override;
    void update_gpu(float* gradInput,OptimizerAdam& optim,int batch_size) override ;
    void forward_gpu(float *input,float* output,int batch_size) override;
    int getInputSize() const override;

    int getOutputSize() const override;
};

struct BatchResult
{
    /* data */
    Matrix input;
    Vector target;
};


class Dataloader{
public:
    Matrix dataset;
    Vector targets;
    int batch_size;
    int len;
    bool shuffle;
    Dataloader(Matrix dataset,Vector targets,int batch_size,bool shuffle);
    BatchResult Get_batch(int index);

};

class MSE{
public:
    float forward(const Vector& predict,const Vector& target);
    Matrix backward(const Matrix& predict,const Vector& target);

    //gpu ----
    // void forward_gpu(float* predict, float* target, float* loss);
    void backward_gpu(float* predict, float* target, float* gradOutput,int batch_size);


};



#endif // MODULE_H