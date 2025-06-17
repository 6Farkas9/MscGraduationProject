<template>
  <div class="twoMenu">
    <div class="two">二级菜单</div>
    <router-view></router-view>
  </div>
</template>

<script setup lang="ts"></script>

<style scoped lang="scss">
.twoMenu {
  width: 90%;
  height: 500px;
  padding: 10px;
  background-color: #e7d8f8;
  border-radius: 10px;
  margin: 0 auto;
  .two {
    margin-bottom: 30px;
  }
}
</style>
