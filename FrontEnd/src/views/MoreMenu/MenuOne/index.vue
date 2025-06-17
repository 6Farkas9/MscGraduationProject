<template>
  <div class="oneMenu">
    <div class="one">一级菜单</div>
    <router-view></router-view>
  </div>
</template>

<script setup lang="ts"></script>

<style scoped lang="scss">
.oneMenu {
  width: 100%;
  font-size: 16px;
  font-weight: bold;
  height: 600px;
  padding: 10px;
  background-color: #bee6f6;
  border-radius: 10px;
  .one {
    margin-bottom: 30px;
  }
}
</style>
