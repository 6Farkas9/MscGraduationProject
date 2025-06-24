<template>
  <div class="platformstats">
    <div class="header">
      <h1>学习平台数据概览</h1>
      <p class="author">—— SZY</p>
    </div>

    <div class="data-cards">
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card" :body-style="{height: '100%'}">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Collection /></el-icon>
              </div>
              <div class="card-text">
                <h3>领域数量</h3>
                <p class="value" v-if="loading.are_num"><span class="loading-text">加载中</span></p>
                <p class="value" v-else>{{ stats.are_num }}</p>
                <p class="description">涵盖的学习领域总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card" :body-style="{height: '100%'}">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><User /></el-icon>
              </div>
              <div class="card-text">
                <h3>学习者数量</h3>
                <p class="value" v-if="loading.lrn_num"><span class="loading-text">加载中</span></p>
                <p class="value" v-else>{{ stats.lrn_num }}</p>
                <p class="description">活跃学习者总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card" :body-style="{height: '100%'}">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Opportunity /></el-icon>
              </div>
              <div class="card-text">
                <h3>学习单元数量</h3>
                <p class="value" v-if="loading.unt_num"><span class="loading-text">加载中</span></p>
                <p class="value" v-else>{{ stats.unt_num }}</p>
                <p class="description">学习单元总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card" :body-style="{height: '100%'}">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Reading /></el-icon>
              </div>
              <div class="card-text">
                <h3>知识点数量</h3>
                <p class="value" v-if="loading.cpt_num"><span class="loading-text">加载中</span></p>
                <p class="value" v-else>{{ stats.cpt_num }}</p>
                <p class="description">系统知识点总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <div class="model-section">
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card" :body-style="{height: '100%'}">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><ChatDotRound /></el-icon>
              </div>
              <div class="card-text">
                <h3>交互数据</h3>
                <p class="value" v-if="loading.ict_num"><span class="loading-text">加载中</span></p>
                <p class="value" v-else>{{ stats.ict_num }}</p>
                <p class="description">系统交互记录总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="18">
          <el-card class="model-card" :body-style="{height: '100%'}">
            <div class="model-content">
              <h3>后台深度学习模型状态</h3>
              <p>上次训练时间: <span v-if="loading.lastTrainingTime" class="loading-text">加载中</span>
                <template v-else>{{ stats.lastTrainingTime }}</template>
              </p>
              <p>模型版本: <span v-if="loading.modelVersion" class="loading-text">加载中</span>
                <template v-else>{{ stats.modelVersion }}</template>
              </p>
              <el-button type="primary" class="train-btn" @click="onTriggerTraining">手动触发训练</el-button>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import {
  Collection,
  User,
  Opportunity,
  Reading,
  ChatDotRound
} from '@element-plus/icons-vue'
import { fetchStats, triggerTraining } from '@/api'

const stats = ref({
  are_num: 0,
  lrn_num: 0,
  unt_num: 0,
  cpt_num: 0,
  ict_num: 0,
  lastTrainingTime: '',
  modelVersion: ''
})

const loading = ref({
  are_num: true,
  lrn_num: true,
  unt_num: true,
  cpt_num: true,
  ict_num: true,
  lastTrainingTime: true,
  modelVersion: true
})

onMounted(async () => {
  try {
    const data = await fetchStats()
    stats.value = data
    // 重置所有加载状态
    Object.keys(loading.value).forEach(key => {
      loading.value[key] = false
    })
  } catch (error) {
    console.error('获取数据失败:', error)
  }
})

const onTriggerTraining = async () => {
  try {
    // 设置相关加载状态
    loading.value.lastTrainingTime = true
    loading.value.modelVersion = true
    
    await triggerTraining()
    const data = await fetchStats()
    stats.value = data
    
    // 重置加载状态
    loading.value.lastTrainingTime = false
    loading.value.modelVersion = false
  } catch (error) {
    console.error('触发训练失败:', error)
    loading.value.lastTrainingTime = false
    loading.value.modelVersion = false
  }
}
</script>

<style scoped>
.platformstats {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 28px;
  color: #333;
  margin-bottom: 10px;
}

.author {
  font-size: 14px;
  color: #999;
  text-align: right;
  margin-right: 20px;
}

.data-cards,
.model-section {
  margin-bottom: 20px;
}

.el-row {
  display: flex;
  flex-wrap: wrap;
}

.el-col {
  display: flex;
  margin-bottom: 20px;
}

.data-card,
.model-card {
  flex: 1;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  display: flex;
  flex-direction: column;
}

.card-content {
  display: flex;
  align-items: center;
  padding: 15px;
  flex: 1;
}

.card-icon {
  font-size: 36px;
  width: 60px;
  height: 60px;
  line-height: 60px;
  text-align: center;
  background: #f0f7ff;
  border-radius: 50%;
  color: #409eff;
  margin-right: 15px;
  flex-shrink: 0;
}

.card-text {
  flex: 1;
}

.card-text h3 {
  font-size: 16px;
  color: #666;
  margin: 0 0 5px 0;
}

.card-text .value {
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin: 0 0 5px 0;
  min-height: 28px;
  display: flex;
  align-items: center;
}

.card-text .description {
  font-size: 12px;
  color: #999;
  margin: 0;
}

.model-content {
  padding: 15px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.model-content h3 {
  margin-top: 0;
  color: #333;
}

.model-content p {
  margin: 10px 0;
  color: #666;
  min-height: 20px;
  display: flex;
  align-items: center;
}

.train-btn {
  margin-top: auto;
  align-self: flex-start;
}

.loading-text {
  color: #a8abb2;
  font-weight: normal;
  position: relative;
  display: inline-flex;
  align-items: center;
}

.loading-text::after {
  content: '...';
  position: absolute;
  left: 100%;
  animation: loadingDots 1.5s infinite steps(4, end);
}

@keyframes loadingDots {
  0%, 20% {
    content: '.';
  }
  40% {
    content: '..';
  }
  60%, 100% {
    content: '...';
  }
}

@media (max-width: 768px) {
  .card-content {
    flex-direction: column;
    text-align: center;
  }
  
  .card-icon {
    margin-right: 0;
    margin-bottom: 10px;
  }
  
  .model-card {
    margin-top: 0;
  }
  
  .el-col {
    margin-bottom: 20px;
  }
}
</style>