//@ts-expect-error mock模拟数据
import Mock from "mockjs"
const userinfo = [
  {
    userId: 2,
    avatar:
      "https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
    username: "admin",
    password: 123456,
    desc: "系统管理员",
    roles: ["系统管理员"],
    buttons: ["cuser.detail", "cuser.user"],
    routes: [
      "document",
      "pagination",
      "edit",
      "home",
      "components",
      "menu-one",
      "moremenu",
      "menu-two",
      "menu-three",
      "menu-three-1",
      "menu-three-2",
      "canvas",
      "news",
      "newsList",
      "newsType",
      "echarts"
    ],
    token:
      "eyJhbGciOiJIUzUxMiIsInppcCI6IkdaSVAifQ.H4sIAAAAAAAAAKtWKi5NUrJSCjAK0A0Ndg1S0lFKrShQsjI0N7I0MjUEUjpKpcWpRZ4pQDEI0y8xNxWoIzElNzNPqRYAHrPX-UEAAAA.u8syR60yvQ_F2-oClEs9xW31Oo1GspbLq-XHOt6xm9UE5vuArmQ63gAiZ8VaQoOtF2snegMe2BlugnoEYDzLeg",
  },
  {
    userId: 2,
    avatar:
      "https://wpimg.wallstcn.com/f778738c-e4f8-4870-b634-56703b4acafe.gif",
    username: "user",
    password: 123456,
    desc: "用户",
    roles: ["用户"],
    buttons: ["cuser.detail", "cuser.user"],
    routes: ["document", "home", "news", "newsList", "components", "edit"],
    token: "user_Token",
  },
]
//用户信息数据
const userinfo_mock = (response: any) => {
  // 获取请求头携带token
  const token = response.body
  //查看用户信息是否包含有次token用户
  const userInfo = userinfo.find((item) => item.token === token)
  //没有返回失败的信息
  if (!userInfo) {
    return { code: 201, data: { message: "获取用户信息失败" } }
  }
  //如果有返回成功信息
  return {
    code: 200,
    message: "获取成功",
    data: userInfo,
  }
}
// 用户登录
const login_mock = (response: any) => {
  //获取请求体携带过来的用户名与密码
  const { username, password } = JSON.parse(response.body)
  //调用获取用户信息函数,用于判断是否有此用户
  const checkUser = userinfo.find(
    (item) => item.username === username && item.password === password,
  )
  //没有用户返回失败信息
  if (!checkUser) {
    return { code: 201, message: "账号或者密码不正确", data: {} }
  }
  //如果有返回成功信息
  const { token } = checkUser
  return { code: 200, data: { token }, message: "登录成功！" }
}
Mock.mock("/api/login", "post", login_mock)
Mock.mock("/api/user", "post", userinfo_mock)
