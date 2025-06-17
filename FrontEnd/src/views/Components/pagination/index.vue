<template>
  <el-row justify="space-between" align="center" gutter="20" class="pagination">
    <el-col :xs="24" :sm="24" :md="24" :lg="16">
      <div class="view">
        <!-- 封装分页器 -->
        <El-pagination
          :config="config"
          :paginationParams="paginationParams"
          @pageChange="pageChange"
          @pageSizeChange="pageSizeChange"
        />
      </div>
    </el-col>
    <el-col :xs="24" :sm="24" :md="24" :lg="8">
      <el-card class="content">
        <div class="card-header">
          <span>操作</span>
        </div>
        <div class="base_config">
          <!-- 基本配置 -->
          <div class="base_item">
            <text>背景色</text>
            <el-switch v-model="config.background" />
          </div>
          <div class="base_item">
            <text>小型分页</text>
            <el-switch v-model="config.small" />
          </div>
          <div class="base_item">
            <text>禁用</text>
            <el-switch v-model="config.disabled" />
          </div>
          <div class="base_item">
            <text>当前页</text>
            <el-input-number
              v-model="paginationParams.page"
              :min="1"
              :max="paginationParams.total / paginationParams.pageSize"
            />
          </div>
          <div class="base_item">
            <text>每页条数</text>
            <el-input-number
              v-model="paginationParams.pageSize"
              :step="10"
              :min="10"
              :max="config.page_sizes[config.page_sizes.length - 1]"
            />
          </div>
        </div>
      </el-card>
    </el-col>
  </el-row>
</template>

<script setup lang="ts">
import { message } from "@/Hooks/Element-plus"
import type {
  paginationConfigType,
  paginationParamsType,
} from "../../../components/ElementPlus_components/El-pagination/type"

const config = reactive(<paginationConfigType>{
  background: true,
  small: false,
  disabled: false,
  layout: "total, sizes, prev, pager, next, jumper",
  page_sizes: [10, 20],
})
const paginationParams = reactive(<paginationParamsType>{
  page: 1,
  pageSize: config.page_sizes[0],
  total: 100,
})
const pageChange = (val: number) => {
  message("info", "当前页变化")
  paginationParams.page = val
}
const pageSizeChange = (val: number) => {
  message("info", "每页条数变化")
  paginationParams.pageSize = val
}
</script>

<style scoped lang="scss">
:deep(.el-input .el-input__wrapper) {
  height: 35px;
  background-color: transparent !important;
}
:deep(.el-input__inner) {
  color: var(--el-text-color-primary) !important;
}
.pagination {
  width: 100%;
  height: 100%;

  .view {
    width: 100%;
    height: 70%;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .content {
    width: 100%;
    height: 80%;
    .card-header {
      padding-bottom: 20px;
      border-bottom: 1px gray solid;
    }
    .base_config {
      width: 100%;
      margin-top: 30px;

      .base_item {
        width: 100%;
        padding: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
    }
  }
}
</style>
