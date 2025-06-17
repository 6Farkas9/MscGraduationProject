/***
 * @常量路由
 */
export const Layout = () => import("@/views/Layout/index.vue")
export const constRouter = [
  {
    path: "/login",
    meta: { title: "登录", icon: "", isShow: false },
    component: () => import("@/views/Login/index.vue"),
  },
  {
    path: "/",
    redirect: "/home",
    name: "layout",
    meta: { title: "", icon: "", isShow: false },
    component: Layout,
    children: [
      {
        path: "/home",
        name: "home",
        meta: { title: "首页", icon: "HomeFilled", isShow: true },
        component: () => import("@/views/Home/index.vue"),
      },
    ],
  },
  {
    path: "/",
    name: "document",
    redirect: "/document",
    meta: { title: "", icon: "", isShow: false },
    component: Layout,
    children: [
      {
        path: "/document",
        name: "document",
        meta: { title: "文档", icon: "Document", isShow: true },
        component: () => import("@/views/Document/index.vue"),
      },
    ],
  },
  {
    path: "/404",
    meta: { title: "404", isShow: false },
    component: () => import("@/views/404/index.vue"),
  },
]
/****
 * @异步路由也叫权限路由
 */
export const asyncRouter = [
  {
    path: "/",
    name: "canvas",
    redirect: "/canvas",
    meta: { title: "", icon: "", isShow: false },
    component: Layout,
    children: [
      {
        path: "/canvas",
        name: "canvas",
        meta: { title: "canvas案例", icon: "EditPen", isShow: true },
        component: () => import("@/views/Canvas/index.vue"),
      },
    ],
  },

  {
    path: "/components",
    name: "components",
    redirect: "/components/pagination",
    meta: { title: "封装组件", icon: "Menu", isShow: true },
    component: Layout,
    children: [
      {
        path: "/components/pagination",
        name: "pagination",
        meta: { title: "分页器 ", icon: "MoreFilled", isShow: true },
        component: () => import("@/views/Components/pagination/index.vue"),
      },
      {
        path: "/components/edit",
        name: "edit",
        meta: { title: "富文本编辑器 ", icon: "Monitor", isShow: true },
        component: () => import("@/views/Components/edit/index.vue"),
      },
    ],
  },

  {
    path: "/moremenu",
    name: "moremenu",
    meta: { title: "多级菜单", icon: "Operation", isShow: true },
    component: Layout,
    children: [
      {
        path: "/moremenu/menu-one",
        name: "menu-one",
        meta: { title: "一级菜单 ", icon: "DArrowRight", isShow: true },
        component: () => import("@/views/MoreMenu/MenuOne/index.vue"),
        children: [
          {
            path: "/moremenu/menu-one/menu-two",
            name: "menu-two",
            meta: { title: "二级菜单 ", icon: "DArrowRight", isShow: true },
            component: () =>
              import("@/views/MoreMenu/MenuOne/MenuTwo/index.vue"),
            children: [
              {
                path: "/moremenu/menu-one/menu-two/menu-three-1",
                name: "menu-three-1",
                meta: {
                  title: "三级菜单-1",
                  icon: "DArrowRight",
                  isShow: true,
                },
                component: () =>
                  import(
                    "@/views/MoreMenu/MenuOne/MenuTwo/MenuThree/index-1.vue"
                  ),
              },
              {
                path: "/moremenu/menu-one/menu-two/menu-three-2",
                name: "menu-three-2",
                meta: {
                  title: "三级菜单-2",
                  icon: "DArrowRight",
                  isShow: true,
                },
                component: () =>
                  import(
                    "@/views/MoreMenu/MenuOne/MenuTwo/MenuThree/index-2.vue"
                  ),
              },
            ],
          },
        ],
      },
    ],
  },
]
/***
 * @任意路由
 */
export const anyRouter = [
  //任意路由
  {
    path: "/:pathMatch(.*)*",
    // name: 'Any',
    meta: { title: "任意路由", isShow: false },
    redirect: "/404",
  },
]
