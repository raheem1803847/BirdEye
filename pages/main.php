
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignIn | BIRDEYE</title>
    
</head>
<body>
    <style>
      @keyframes animate
{
    
    0%,100%{
        background-image: url("1.jpeg");
    }

    75%{
        background-image: url("2.jpeg");

    }
    
    50%{
        background-image: url( "3.jpeg");

    }
    
    25%{
        background-image: url("3.jpeg");

    }
}
    .login-box{
        width: 280px;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%,-50%);
        color: white;
      
      }
      .login-box h1{
        float:left;
        font-size: 40px;
        border-bottom: 6px solid #4caf50 ;
        margin-bottom: 50px;
        padding: 13px 0;
        
      }
      .textbox{
        width: 100%;
        overflow: hidden;
        font-size: 20px;
        padding: 8px 0;
        margin: 8px 0;
        border-bottom: 1px solid #4caf50 ;
      }
      .textbox i{
        width: 26px;
        float: left;
        text-align: center;
      }
      .textbox input{
        border:none;
        outline:none;
        background: none;
        color: white;
        font-size: 18px;
        width: 80%;
        float: left;
        margin:0 10px
      }
      .signin{
        width:100%;
        background:none;
        border: 2px solid #4caf50;
        color:white;
        padding:5px;
        font-size: 18px;
        cursor:pointer;
        margin: 12px 0;
      }
      .Signin-container{
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100vh;
        animation: animate 16s ease-in-out infinite ;
        background-size: cover;
        filter: brightness(30%);
      
}

    </style>
    <div class= "Signin-container">
    </div>
    <form action="test.php">
        <div class="login-box">
            <h1>Login</h1>
        <div class="textbox">
        <i class="fa fa-user" aria-hidden="true"></i>
            <input type="text" placeholder="Username" name="user_name" value="" required>
        </div>
        <div class="textbox">
        <i class="fa fa-lock" aria-hidden="true"></i>
            <input type="password" placeholder="Password" name="password" value=""required>
        </div>
        <input class="signin" type="submit" name="signin" value="Sign In">
        <script>
            
          </script>
    </form>
</body>
</html>


